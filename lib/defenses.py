import torch
import copy

import lib.perturbations as perturbations

class Defense:
    def __init__(self, target_model):
        self.target_model = target_model

class SmoothLLM(Defense):

    """SmoothLLM defense.
    
    Title: SmoothLLM: Defending Large Language Models Against 
                Jailbreaking Attacks
    Authors: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas
    Paper: https://arxiv.org/abs/2310.03684
    """

    def __init__(self, 
        target_model,
        # perturbation_fn,
        pert_type,
        pert_pct,
        num_copies,
        test_prefixes,
    ):
        super(SmoothLLM, self).__init__(target_model)
        
        self.test_prefixes = test_prefixes
        # self.perturbation_fn = perturbation_fn
        self.num_copies = num_copies

        self.perturbation_fn = vars(perturbations)[pert_type](
            q=pert_pct
        )

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
            batch_outputs = self.target_model(
                batch=batch, 
                max_new_tokens=prompt.max_new_tokens
            )

            all_outputs.extend(batch_outputs)
            torch.cuda.empty_cache()
            
        # Check whether the outputs jailbreak the LLM
        return [self.is_jailbroken(s) for s in all_outputs]


        