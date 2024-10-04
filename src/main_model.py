import torch
from torch import nn
import warnings
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from transformers import (RagModel,
                          RagRetriever, 
                          RagSequenceForGeneration, 
                          RagTokenForGeneration, 
                          RagConfig, 
                          RagPreTrainedModel, 
                          AutoModel, 
                          AutoModelForSeq2SeqLM,
                          T5ForConditionalGeneration, 
                          T5Config)

from transformers.models.rag.modeling_rag import RetrievAugLMMarginOutput, RetrievAugLMOutput
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import (
                                        BaseModelOutput,
                                        BaseModelOutputWithPastAndCrossAttentions,
                                        Seq2SeqLMOutput,
                                        Seq2SeqModelOutput,
                                        Seq2SeqQuestionAnsweringModelOutput,
                                        Seq2SeqSequenceClassifierOutput,
                                    )

@dataclass
class CustomRetrievAugLMMarginOutput(RetrievAugLMMarginOutput):
    loss: Optional[torch.FloatTensor] = None    
    loss_sub_1: torch.FloatTensor = None
    loss_sub_2: torch.FloatTensor = None
    loss_sub_3: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    prob_true: torch.FloatTensor = None ###
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class CustomRagModel(RagPreTrainedModel):
    def __init__(
        self,
        config = None,
        question_encoder = None,
        generator = None,
        retriever = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config)
        '''
        if question_encoder is None:

            question_encoder = AutoModel.from_config(config.question_encoder)
        '''
        if generator is None:

            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        #self.question_encoder = question_encoder
        self.generator = generator

        #self.ctx_encoder = question_encoder
        self.context_encoder_training = False
        
##################################################################################################
##################################################################################################

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        encoder_outputs = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        past_key_values = None,
        doc_scores = None,
        context_input_ids = None,
        context_attention_mask=None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        output_retrieved = None,
        n_docs = None,
    ):

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )
                question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    return_tensors="pt",
                )
                if self.context_encoder_training:
                    (
                        context_input_ids,
                        context_attention_mask,
                        retrieved_doc_embeds,
                        retrived_doc_input_ids,
                        retrived_doc_attention_mask,
                        retrieved_doc_ids,
                    ) = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["tokenized_doc_ids"],
                        retriever_outputs["tokenized_doc_attention_mask"],
                        retriever_outputs["doc_ids"],
                    )

                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    retrived_doc_input_ids = retrived_doc_input_ids.to(input_ids)
                    retrived_doc_attention_mask = retrived_doc_attention_mask.to(input_ids)
                    retrieved_doc_embeds = self.ctx_encoder(
                        retrived_doc_input_ids, attention_mask=retrived_doc_attention_mask, return_dict=True
                    ).pooler_output
                    retrieved_doc_embeds = retrieved_doc_embeds.view(
                        -1, n_docs, question_encoder_last_hidden_state.shape[1]
                    )  # reshaping

                    # compute doc_scores involving ctx_encoder
                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                else:
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["doc_ids"],
                    )

                    # set to correct device
                    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    # compute doc_scores
                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)
            else:
                assert context_input_ids is not None, (
                    "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can"
                    " set a retriever using the `set_retriever(...)` function."
                )
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."
        
        assert (doc_scores.shape[1] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
            f" The first dimension of `doc_scores.shape` should be a multiple of `n_docs`={n_docs}, but is"
            f" {doc_scores.shape[1]}."
        )

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)
        
        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
            generator_cross_attentions=gen_outputs.cross_attentions,
        )
    
class CustomRagSequenceForGeneration(RagSequenceForGeneration):
    def __init__(
        self,
        config = None,
        question_encoder = None,
        generator = None,
        retriever = None,
        quality_estimator = None,
        loss_option = "no",
        loss_weight = 1,
        **kwargs,
    ):
        super().__init__(config, question_encoder, generator, retriever, **kwargs)

        # instantiate model
        self.rag = CustomRagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

        self.loss_weight = loss_weight
        self.loss_option = loss_option

        if quality_estimator is not None:
            self.quality_estimator = quality_estimator
            self.doc_cal = "calculate"                    
        else:
            self.doc_cal = "none"
            
        print(f"document calculation mode is {self.doc_cal}")

    def softmax_scaling_per_batch(self, data, dim=1):
        # Apply softmax to scale the data into a [0, 1] range
        softmax_scaled = torch.nn.functional.softmax(data, dim=dim)

        normalized = softmax_scaled * 2 - 1

        positive_sum = torch.sum(torch.where(normalized > 0, normalized, torch.zeros_like(normalized)), dim=dim, keepdim=True)
        negative_sum = torch.sum(torch.where(normalized < 0, normalized, torch.zeros_like(normalized)), dim=dim, keepdim=True)

        ratio = torch.abs(positive_sum) / torch.abs(negative_sum)
        adjusted_negative = torch.where(normalized < 0, normalized * ratio, normalized)

        return adjusted_negative
    
    def min_max_scaling_per_batch(self, data, dim=1):
        min_vals = torch.min(data, dim=dim, keepdim=True).values
        max_vals = torch.max(data, dim=dim, keepdim=True).values
        normalized = (data - min_vals) / (max_vals - min_vals)
        
        normalized = normalized * 2 - 1

        positive_sum = torch.sum(torch.where(normalized > 0, normalized, torch.zeros_like(normalized)), dim=dim, keepdim=True)
        negative_sum = torch.sum(torch.where(normalized < 0, normalized, torch.zeros_like(normalized)), dim=dim, keepdim=True)

        ratio = torch.abs(positive_sum) / torch.abs(negative_sum)

        adjusted_negative = torch.where(normalized < 0, normalized * ratio, normalized)

        return adjusted_negative

    def probability_to_logit(self, probabilities, eps=1e-5):
        probabilities = torch.clamp(probabilities, eps, 1 - eps)
        logits = torch.log(probabilities / (1 - probabilities))
        return logits
    
    def calculate_qc_docscore(self, qe_input_ids, qe_attention_mask, n_docs, temperature = 1.0):
        
        size = qe_input_ids.shape[0]
        decoder_input_ids = torch.tensor([[0]] * size)
        
        qe_input_ids_device = qe_input_ids.device
        decoder_input_ids = decoder_input_ids.to(qe_input_ids_device)
        
        outputs, logit = self.quality_estimator(input_ids=qe_input_ids, attention_mask=qe_attention_mask, decoder_input_ids=decoder_input_ids)
 
        first_token_logits = logit[:, 0, :]
        
        probabilities = torch.softmax(first_token_logits, dim=-1)

        prob_true = probabilities[:, 1176].unsqueeze(0) 
        prob_false = probabilities[:, 6136].unsqueeze(0) 

        prob_true_raw = prob_true.clone()
        prob_false_raw = prob_false.clone()
        prob_true_raw_mean = prob_true_raw.mean()
        prob_false_raw_mean = prob_false_raw.mean()
        
        prob_sum = prob_true + prob_false 

        prob_true = prob_true / prob_sum 
        prob_false = prob_false / prob_sum 

        logit_true = self.probability_to_logit(prob_true) 
        logit_false = self.probability_to_logit(prob_false) 
        
        doc_score_calculated = logit_true.view(-1, n_docs)
        doc_score_calculated = doc_score_calculated / temperature
        prob_true = prob_true.view(-1, n_docs)
        doc_neg_score_calculated = logit_false.view(-1, n_docs)
        
        prob_false_raw = prob_false_raw.view(-1, n_docs) 
        prob_true_raw = prob_true_raw.view(-1, n_docs) 

        return doc_score_calculated, prob_true, prob_true_raw_mean, prob_false_raw_mean, prob_false_raw, prob_true_raw
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        encoder_outputs = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        past_key_values = None,
        context_input_ids = None,
        context_attention_mask = None,
        qe_input_ids = None,
        qe_attention_mask = None,
        doc_scores = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        output_retrieved = None,
        exclude_bos_score = None,
        reduce_loss = None,
        labels = None,
        n_docs = None,
        **kwargs,  # needs kwargs for generation
    ):

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False
            
        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        prob_true = None
        if self.doc_cal == "calculate" and qe_input_ids is not None:            
            doc_score_calculated, prob_true, prob_true_raw_mean, prob_false_raw_mean, prob_false_raw, prob_true_raw = self.calculate_qc_docscore(qe_input_ids, qe_attention_mask, n_docs) 
            outputs.doc_scores = doc_score_calculated        

        loss = None
        if labels is not None:
            if self.doc_cal == "calculate" and qe_input_ids is not None:
                loss, _ll = self.get_nll(
                        outputs.logits,
                        doc_score_calculated,
                        labels, 
                        reduce_loss=reduce_loss,
                        epsilon=self.config.label_smoothing,
                        exclude_bos_score=exclude_bos_score,
                        n_docs=n_docs,
                    )                
            else:     
                loss, _ll = self.get_nll(
                        outputs.logits,
                        outputs.doc_scores,
                        labels,
                        reduce_loss=reduce_loss,
                        epsilon=self.config.label_smoothing,
                        exclude_bos_score=exclude_bos_score,
                        n_docs=n_docs,
                    )

        if self.loss_option == "yes" and qe_input_ids is not None:
            _ll = _ll.detach()

            kl_loss = nn.KLDivLoss(reduction="batchmean")
            _ll_softmax = torch.nn.functional.softmax(_ll, dim=-1)
            doc_score_log_softmax = torch.nn.functional.log_softmax(doc_score_calculated, dim=-1)

            loss_sub_1 = kl_loss(doc_score_log_softmax, _ll_softmax)
            loss_sub_2 = 0 # delete loss

            loss_add_3 = 1 - (prob_true_raw_mean + prob_false_raw_mean)            
            
            loss = loss + self.loss_weight * (loss_sub_1) + 1.0 * loss_add_3 
        else:
            loss_sub_1 = None
            loss_sub_2 = None
            loss_add_3 = None

        return CustomRetrievAugLMMarginOutput(
            loss=loss,
            loss_sub_1 = loss_sub_1,
            loss_sub_2 = loss_sub_2,
            loss_sub_3 = loss_add_3,
            logits=outputs.logits,
            doc_scores=outputs.doc_scores,
            prob_true = prob_true, # added
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids = None,
        attention_mask = None,
        context_input_ids = None,
        context_attention_mask = None,
        qe_input_ids = None,
        qe_attention_mask = None,
        doc_scores = None,
        do_deduplication = None,  # defaults to True
        num_return_sequences = None,  # defaults to 1
        num_beams = None,  # defaults to 1
        n_docs = None,
        **model_kwargs):

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        assert (
            input_ids is not None or context_input_ids is not None
        ), " At least one of input_ids or context_input_ids must be given"

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )["context_input_ids"]

            # set to correct device
            context_input_ids = context_input_ids.to(input_ids)
        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams
        model_kwargs["attention_mask"] = None

        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs
        
        for index in range(batch_size):       
            
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)
            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            if do_deduplication:
                # do_deduplication, max_output_len
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))
            
            num_candidates = output_sequences.shape[
                0
            ]  # after deduplication, this number can be less than n_docs*n_beam
            ########################################################################################################
            if qe_input_ids is not None:
                individual_qe_ids = qe_input_ids[index * n_docs : (index + 1) * n_docs]
                individual_qe_ids = individual_qe_ids.repeat(num_candidates, 1)
            else:
                individual_qe_ids = None
                
            if qe_attention_mask is not None:
                individual_qe_attention_mask = qe_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_qe_attention_mask = individual_qe_attention_mask.repeat(num_candidates, 1)
            else:
                individual_qe_attention_mask = None
            ########################################################################################################
            
            # then, run model forwards to get nll scores:
            if input_ids is not None:
                new_input_ids = input_ids[index : index + 1].repeat(num_candidates, 1)
                outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            else:  # input_ids is None, need context_input_ids/mask and doc_scores
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

                individual_input_ids = generator_input_ids.repeat(
                    num_candidates, 1
                )  # (num_candidates*n_docs, max_len)

                individual_attention_mask = context_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)

                individual_doc_scores = doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]
                
                outputs = self(
                    context_input_ids=individual_input_ids,
                    context_attention_mask=individual_attention_mask,
                    qe_input_ids=individual_qe_ids,
                    qe_attention_mask=individual_qe_attention_mask,                       
                    doc_scores=individual_doc_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                    n_docs = n_docs # add this line
                )
                    
            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(output_sequences[top_cand_inds])
            # dim add
            for i in range(len(hypos)):
                if hypos[i].dim() == 1:
                    hypos[i] = hypos[i].unsqueeze(0)

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)
    
    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # bos_token_id is None for T5
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size

        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        
        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # calculate loss
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

        ##########################################################

        _rag_logprobs = torch.cat([first_token_scores, second_token_scores, remainder], dim=2) # revised

        # calculate nll
        _ll = _rag_logprobs.gather(dim=-1, index=target)
        _smooth_obj = _rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        _ll, _smooth_obj = _mask_pads(_ll, _smooth_obj)

        # sum over tokens, exclude bos while scoring
        _ll = _ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else _ll.sum(2) # _ll : batch_size, n_docs
        # _ll = _ll.logsumexp(1)  # logsumexp over docs

        ##########################################################

        return loss, _ll
        