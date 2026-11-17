"""CyberDog: beam search over only the tokens VAMOS answers with.

VAMOS answers with 10 <locNNNN> tokens and EOS, nothing else. HF's
generate() still scores all 257,216 tokens every step: the output head
(2304 x 257216, 1.2 GB bf16) is ~23% of the weights read per step (~29 ms of
~177 ms at 25W on the J4012), and beam search then runs log_softmax/topk over
5 x 257,216 scores (~380 ms a call). Here the head is cut to the 1,025 rows
(<loc0000>..<loc1023> + EOS) and beam search runs over those.

The model's forward is HF's own (PaliGemmaModel, DynamicCache, the same
position_ids and mask generate() passes). Prefill runs once at batch 1 and
its cache is repeated to every beam, as install_shared_prefill does.
PaliGemma applies no logit softcap, so each kept token's score is exactly the
full head's. What changes: log_softmax normalises over the 1,025 tokens only
(probability on other tokens is dropped), and the answer can only contain
<loc> tokens and EOS.

Beam search: a line-by-line port of transformers 5.x _beam_search for batch 1
(length_penalty 1.0, early_stopping False, no logits processors, one EOS).
With subset=False (the full head) it must give generate()'s paths exactly:
that is how the port is checked.

goal tail (generate(goal=...), vlm_server --goal_tail): VAMOS's last point is
the goal the prompt names, then EOS -- 80/80 paths on the J4012 (16 prompts x
5 beams). Decoding it costs 3 of the 10 decode steps (~150 ms each at 25W,
GPU-bound). With a goal, the model decodes the first GOAL_TAIL_AT tokens (4
points) and every running beam is finished with <goal x><goal y><EOS> at
log-probability 0, scored by the same length-normalised rule as a real EOS.
What it changes: a path that would not have ended at the goal now does.

Guard: the tail is used only when the answer still has VAMOS's shape. If the
<loc> head is off (subset=False), there is no room for the tail, or the model
ended any kept beam early (EOS inside the first 4 points), the call decodes in
full instead and the fallback is counted (tail_used / tail_fallback,
last_tail). Any early end, not only one that would rank: the tail scores its
3 tokens at log-probability 0, so it would rank the full paths a little too
well against an early end. A retrained model with another answer
shape then shows up in the log, not as quietly wrong paths.
"""
import logging

import torch

logger = logging.getLogger(__name__)

GOAL_TAIL_AT = 8          # generated tokens the model decodes before the goal tail: 4 points


class LocHeadDecoder:
    def __init__(self, model, tokenizer, num_beams=5, subset=True):
        self.model = model
        self.num_beams = num_beams
        self.tail_used = 0                                     # goal tail: calls it ended
        self.tail_fallback = 0                                 # calls decoded in full instead
        self.last_tail = None                                  # "used", "fallback" or None
        gc = model.generation_config
        eos = gc.eos_token_id
        eos = eos if isinstance(eos, int) else (eos[0] if eos and len(eos) == 1 else None)
        if eos is None:
            raise ValueError(f"expects exactly one EOS token, got {gc.eos_token_id}")
        self.eos = eos
        self.pad = gc.pad_token_id
        head = model.get_output_embeddings().weight          # [vocab, hidden]
        if subset:
            locs = tokenizer.convert_tokens_to_ids([f"<loc{i:04d}>" for i in range(1024)])
            if tokenizer.unk_token_id in locs:
                raise ValueError("tokenizer has no <locNNNN> tokens")
            ids = torch.tensor([eos] + locs, dtype=torch.long, device=head.device)
            self.ids = ids                                      # head row -> token id
            self.head = head.index_select(0, ids).contiguous()  # [1025, hidden], ~4.7 MB
        else:
            self.ids = None
            self.head = head
        self.V = self.head.shape[0]

    def _fallback(self, why):
        self.tail_fallback += 1
        self.last_tail = "fallback"
        logger.warning(f"goal tail not used ({why}): decoding in full "
                       f"[{self.tail_fallback} fallback(s), {self.tail_used} tail(s) so far]")

    def _logits(self, hidden):
        return torch.nn.functional.linear(hidden[:, -1, :], self.head)

    @torch.no_grad()
    def generate(self, model_input, max_new_tokens=20, num_return_sequences=None, goal=None):
        """Beam search; `goal` (x token id, y token id) ends every path there (goal tail)."""
        n = self.num_beams
        nret = num_return_sequences or n
        self.last_tail = None
        if goal is not None and self.ids is None:
            goal = None                                        # full head: answer shape not guaranteed
            self._fallback("the <loc> head is off")
        ids = model_input["input_ids"]
        if ids.shape[0] != 1:
            raise ValueError("batch 1 only")
        P = ids.shape[1]
        max_length = P + max_new_tokens
        dev = ids.device
        V = self.V
        body = self.model.model                                # PaliGemmaModel: hidden states, no head

        # prefill at batch 1, cache repeated to the beams
        kw = {k: v for k, v in model_input.items() if k != "labels"}
        out = body(**kw, use_cache=True)
        cache = out.past_key_values
        cache.batch_repeat_interleave(n)
        logits = self._logits(out.last_hidden_state).expand(n, -1)
        del out

        beams_to_keep = 2 * n                                  # max(2, 1 + n_eos) * num_beams, one EOS
        top_num_beam_mask = torch.cat((torch.ones(n, dtype=torch.bool),
                                       torch.zeros(n, dtype=torch.bool))).to(dev)
        fill = self.pad if self.pad else self.eos              # HF: `pad_token_id or eos_token_id[0]`
        running_sequences = torch.full((1, n, max_length), fill, dtype=torch.int64, device=dev)
        running_sequences[:, :, :P] = ids.expand(n, -1)
        sequences = running_sequences.clone()
        running_beam_scores = torch.zeros((1, n), dtype=torch.float, device=dev)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = torch.full((1, n), -1e9, dtype=torch.float, device=dev)
        is_sent_finished = torch.zeros((1, n), dtype=torch.bool, device=dev)
        unsat = torch.ones((1, 1), dtype=torch.bool, device=dev)
        running_beam_indices = torch.full((1, n, max_length - P), -1, dtype=torch.int32, device=dev)
        beam_indices = running_beam_indices.clone()
        cur_len = P

        def gather(t, idx):
            while idx.dim() < t.dim():
                idx = idx.unsqueeze(-1)
            return torch.take_along_dim(t, idx, dim=1)

        while True:
            log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
            log_probs = (log_probs.view(1, n, V) + running_beam_scores[:, :, None]).reshape(1, n * V)

            topk_log_probs, topk_indices = torch.topk(log_probs, k=beams_to_keep)
            topk_beam = topk_indices // V
            topk_running_beam_indices = gather(running_beam_indices, topk_beam)
            topk_running_sequences = gather(running_sequences, topk_beam)
            topk_ids = topk_indices % V
            if self.ids is not None:
                topk_ids = self.ids[topk_ids]                  # head row -> real token id
            topk_running_sequences[:, :, cur_len] = topk_ids
            topk_running_beam_indices[:, :, cur_len - P] = topk_beam.to(torch.int32)

            hits = (topk_ids == self.eos) | (cur_len + 1 >= max_length)

            topk_running_log_probs = topk_log_probs + hits.to(torch.float32) * -1.0e9
            nxt = torch.topk(topk_running_log_probs, k=n)[1]
            running_sequences = gather(topk_running_sequences, nxt)
            running_beam_scores = gather(topk_running_log_probs, nxt)
            running_beam_indices = gather(topk_running_beam_indices, nxt)

            just_finished = hits & top_num_beam_mask[None, :]
            scored = topk_log_probs / float(cur_len + 1 - P)              # length_penalty 1.0
            scored = scored + (~unsat).to(torch.float32) * -1.0e9         # early_stopping False
            scored = scored + (~just_finished) * -1.0e9
            merged_seq = torch.cat((sequences, topk_running_sequences), dim=1)
            merged_scores = torch.cat((beam_scores, scored), dim=1)
            merged_idx = torch.cat((beam_indices, topk_running_beam_indices), dim=1)
            merged_fin = torch.cat((is_sent_finished, just_finished), dim=1)
            top = torch.topk(merged_scores, k=n)[1]
            sequences = gather(merged_seq, top)
            beam_scores = gather(merged_scores, top)
            beam_indices = gather(merged_idx, top)
            is_sent_finished = gather(merged_fin, top)

            beam_idx = running_beam_indices[0, :, cur_len - P].long()
            cur_len += 1

            if goal is not None and cur_len - P == GOAL_TAIL_AT and cur_len + 3 > max_length:
                goal = None
                self._fallback(f"max_new_tokens {max_new_tokens} leaves no room for the tail")
            if goal is not None and cur_len - P == GOAL_TAIL_AT and bool(is_sent_finished.any()):
                goal = None                                    # not the 5-point shape the tail assumes
                self._fallback("a path ended before 4 points")
            if goal is not None and cur_len - P == GOAL_TAIL_AT:
                # every running beam ends <goal x><goal y><EOS>; merged with the
                # finished ones as a real EOS would be (score / generated length)
                tail = torch.tensor([goal[0], goal[1], self.eos], dtype=torch.int64, device=dev)
                seq = running_sequences.clone()
                seq[:, :, cur_len:cur_len + 3] = tail
                idx = running_beam_indices.clone()
                idx[:, :, cur_len - P:cur_len - P + 3] = torch.arange(
                    n, dtype=torch.int32, device=dev)[None, :, None]
                score = running_beam_scores / float(cur_len + 3 - P)
                merged_seq = torch.cat((sequences, seq), dim=1)
                merged_scores = torch.cat((beam_scores, score), dim=1)
                merged_idx = torch.cat((beam_indices, idx), dim=1)
                top = torch.topk(merged_scores, k=n)[1]
                sequences = gather(merged_seq, top)
                beam_indices = gather(merged_idx, top)
                self.tail_used += 1
                self.last_tail = "used"
                break

            best_possible = running_beam_scores[:, :1] / float(cur_len - P)
            worst_finished = torch.where(is_sent_finished,
                                         torch.min(beam_scores, dim=1, keepdim=True)[0], -1.0e9)
            unsat = unsat & torch.any(best_possible > worst_finished, dim=-1, keepdim=True)
            if not bool(torch.any(unsat) & ~torch.all(hits)):             # one sync a step, as HF
                break

            cache.reorder_cache(beam_idx)
            # the inputs generate() passes on a decode step
            tok = running_sequences[0, :, cur_len - 1:cur_len]
            pos = torch.full((n, 1), cur_len, dtype=torch.long, device=dev)   # 1-indexed position
            mask = torch.ones((n, cur_len), dtype=torch.long, device=dev)
            out = body(input_ids=tok, position_ids=pos, attention_mask=mask,
                       past_key_values=cache, use_cache=True)
            logits = self._logits(out.last_hidden_state)
            del out

        if goal is not None and self.last_tail is None:
            self._fallback("the search ended before 4 points")      # answered in full anyway

        sequences = sequences[0, :nret]
        beam_indices = beam_indices[0, :nret]
        gen_len = int((beam_indices + 1).bool().sum(dim=1).max())
        return sequences[:, :P + gen_len]
