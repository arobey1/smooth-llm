import json
import copy
import pandas as pd
from fastchat.model import get_conversation_template

from llm_attacks import (
    AttackPrompt,
    MultiPromptAttack,
    PromptManager
)

class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens

    def perturb(self, perturbation_fn):
        perturbed_prompt = perturbation_fn(self.perturbable_prompt)
        self.full_prompt = self.full_prompt.replace(
            self.perturbable_prompt,
            perturbed_prompt
        )
        self.perturbable_prompt = perturbed_prompt

class Attack:
    def __init__(self, logfile):
        self.logfile = logfile

class GCG(Attack):

    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on 
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, logfile, workers):
        super(GCG, self).__init__(logfile)

        self.workers = workers
        self.managers = {
            "AP": AttackPrompt,
            "PM": PromptManager,
            "MPA": MultiPromptAttack
        }

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        self.goals = log['goal']
        self.targets = log['target']
        self.controls = log['controls']

        self.prompts = [
            self.create_prompt(g, c, t)
            for (g, c, t) in zip(self.goals, self.controls, self.targets)
        ]

    def create_prompt(self, goal, control, target, max_new_len=100):
        """Create GCG prompt."""

        # Load GCG attack prompt
        attack = self.managers['MPA'](
            [goal], 
            [target],
            self.workers,
            control,
            test_prefixes=[],
            logfile=None,
            managers=self.managers,
        )
        max_new_tokens = max(
            [p.test_new_toks for p in attack.prompts[0]._prompts][0],
            max_new_len
        )
        full_prompt = [p.eval_str for p in attack.prompts[0]._prompts][0]
        
        start_index = full_prompt.find(goal)
        end_index = full_prompt.find(control) + len(control)
        perturbable_prompt = full_prompt[start_index:end_index]
        
        return Prompt(
            full_prompt, 
            perturbable_prompt, 
            max_new_tokens
        )

class PAIR(Attack):

    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, 
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, logfile, workers):
        super(PAIR, self).__init__(logfile)

        self.workers = workers

        df = pd.read_pickle(logfile)
        jailbreak_prompts = df['jailbreak_prompt'].to_list()
        
        self.prompts = [
            self.create_prompt(prompt)
            for prompt in jailbreak_prompts
        ]
        
    def create_prompt(self, prompt):
        
        template = copy.deepcopy(self.workers[0].conv_template)
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None) 
        
        full_prompt = template.get_prompt()

        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=100
        )