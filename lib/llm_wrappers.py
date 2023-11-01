import torch
import copy

class SmoothLLM:

    """Forward pass through SmoothLLM."""

    def __init__(self, 
        workers,
        perturbation_fn,
        num_copies,
        test_prefixes,
    ):

        self.workers = workers
        self.LLM = LLM(
            model=workers[0].model,
            tokenizer=workers[0].tokenizer
        )

        self.test_prefixes = test_prefixes
        self.perturbation_fn = perturbation_fn
        self.num_copies = num_copies

    def is_jailbroken(self, s):
        """Returns True if a prompt results in a jailbreak; False otherwise."""

        return not any([
            prefix in s for prefix in self.test_prefixes
        ])

    @torch.no_grad()
    def __call__(self, prompt, batch_size=64, max_new_len=100):

        all_inputs = []
        for _ in range(self.num_copies):
            prompt_copy = copy.deepcopy(prompt)
            prompt_copy.perturb(self.perturbation_fn)
            all_inputs.append(prompt_copy.full_prompt)

        # Iterate each batch of inputs
        all_outputs = []
        for i in range(self.num_copies // batch_size + 1):

            # Get the current batch of inputs
            batch = all_inputs[i * batch_size:(i+1) * batch_size]

            # Run a forward pass through the LLM for each perturbed copy
            batch_outputs = self.LLM(
                batch=batch, 
                max_new_tokens=prompt.max_new_tokens
            )

            all_outputs.extend(batch_outputs)
            torch.cuda.empty_cache()
            
        # Check whether the outputs jailbreak the LLM
        return [
            self.is_jailbroken(s) for s in all_outputs
        ]

class LLM:

    """Forward pass through a LLM."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'

    def __call__(self, batch, max_new_tokens=100):

        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # Forward pass through the LLM
        try:
            outputs = self.model.generate(
                batch_input_ids, 
                attention_mask=batch_attention_mask, 
                max_new_tokens=max_new_tokens
            )
        except RuntimeError:
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]

        return batch_outputs