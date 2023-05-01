import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class MLPAgent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], dropout=0):
        super().__init__()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i, j in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(nn.ReLU())
            # layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers[:-1])
        
    def forward(self, x):
        return self.net(x)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    def __init__(self, block_size, **kwargs):
        if kwargs['model_type'] in ['s+a', 's+a+g', 's+a+g+p']:
            self.block_size = block_size * 2
        if kwargs['model_type'] in ['s', 's+g', 's+g+p']:
            self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.len_goal_type = len(config.goal_type)
        self.variant = config.variant
        if self.variant in ['v5', 'v6', 'v7']:
            self.mask[:,:,:self.len_goal_type] = 1.0

    def forward(self, x, layer_past=None, random_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        if self.variant in ['v5', 'v6', 'v7']:
            assert random_mask is not None
            att = att.masked_fill(random_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Blocks(nn.Module):
    def __init__(self, *args, config=''):
        super().__init__()
        self.blocks = args 
        self.variant = config.variant
        self.n_head = config.n_head
        self.len_goal_type = len(config.goal_type)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        if self.variant in ['v5', 'v6', 'v7']:
            if mask is None:
                B, T, _ = x.shape
                r = torch.randint(0, (T - self.len_goal_type) // 2, [B])[:, None]
                r = r * 2 + self.len_goal_type
                m = torch.arange(0, T).repeat(B, 1) > r
                random_mask = torch.zeros(B, self.n_head, T, T, dtype=bool)
                random_mask[:, :, :self.len_goal_type, :] = \
                    m[:, None, None, :].repeat(1, self.n_head, self.len_goal_type, 1)
                random_mask = random_mask.cuda()
            else:
                random_mask = mask.cuda()
        else:
            random_mask = None
        output = []
        for block in self.blocks:
            x = block(x, random_mask=random_mask)
            output.append(x)
        return x, output

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, random_mask=None):
        x = x + self.attn(self.ln1(x), random_mask=random_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, 
            config, 
            state_dim=-1, 
            action_dim=-1, 
            variant='', 
            progress_monitor=0,
            task=None):
        super().__init__()

        self.config = config
        assert state_dim > 0 and action_dim > 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = config.model_type
        self.goal_type = config.goal_type
        self.variant = variant
        self.task = task
        self.progress_monitor = progress_monitor
        self.goal_loss_type = config.goal_loss_type \
            if hasattr(config, 'goal_loss_type') else ''
        config.variant = variant

        assert self.task is not None

        if self.model_type in ['s', 's+g', 's+g+p']:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, config.block_size, config.n_embd))
        if self.model_type in ['s+a', 's+a+g', 's+a+g+p']:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, config.block_size//2, config.n_embd))
        if '+g' in self.model_type:
            assert self.goal_type != '' and \
                np.all([ord('z') >= ord(g) >= ord('a') for g in self.goal_type])
            self.goal_pos_emb = nn.Parameter(
                torch.zeros(1, len(self.goal_type), config.n_embd))
            config.block_size += len(self.goal_type)  # Add the extra goal state.
        self.global_pos_emb = nn.Parameter(
            torch.zeros(1, config.max_timestep, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer.
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.blocks_wrapper = Blocks(*self.blocks, config=config)
        
        # Decoder head.
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.action_head = MLPAgent(config.n_embd, action_dim, hidden_dims=[256,256])

        if '+p' in self.model_type:
            # V1
            self.goal_pred = MLPAgent(
                state_dim, state_dim * len(self.goal_type), hidden_dims=[256,256])
            # V2
            # self.goal_pred1 = MLPAgent(state_dim, state_dim, hidden_dims=[256,256])
            # self.goal_pred2 = MLPAgent(state_dim, state_dim, hidden_dims=[256,256])

        self.block_size = config.block_size
        if '+g' in self.model_type:
            self.block_size -= len(self.goal_type)  # Remove the extra goal state.
        self.apply(self._init_weights)

        # Action embeddings.
        if self.model_type in ['s+a', 's+a+g', 's+a+g+p']:
            self.action_encoder = nn.Sequential(
                nn.Linear(self.action_dim, 256), 
                nn.ReLU(),
                nn.Linear(256, config.n_embd), 
            )
        
        # State embeddings.
        self.state_encoder = MLPAgent(
            self.state_dim, config.n_embd, hidden_dims=[256])
        if self.variant == 'v4':
            self.state_encoder_v2 = MLPAgent(
                self.state_dim, config.n_embd, hidden_dims=[256])
        if self.variant == 'v5':
            if len(self.goal_loss_type) > 1:
                self.state_decoder_list = [MLPAgent(config.n_embd, self.state_dim,
                    hidden_dims=[]).cuda() for _ in self.goal_loss_type]
            else:
                self.state_decoder = MLPAgent(
                    config.n_embd, self.state_dim, hidden_dims=[])
        if self.variant in ['v6', 'v8', 'v10']:
            if len(self.goal_loss_type) > 1:
                self.state_decoder_list = [MLPAgent(config.n_embd, self.state_dim,
                    hidden_dims=[256]).cuda() for _ in self.goal_loss_type]
            else:
                self.state_decoder = MLPAgent(
                    config.n_embd, self.state_dim, hidden_dims=[256])
        if self.variant in ['v7', 'v9', 'v11']:
            self.state_decoder = MLPAgent(
                config.n_embd, self.state_dim, hidden_dims=[256, 256])

        if self.progress_monitor > 0:
            if self.task == 'StackCube-v0':
                n_progress_class = 3
            self.progress_classifier = MLPAgent(
                config.n_embd, n_progress_class, 
                hidden_dims=[256] * self.progress_monitor)

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        if '+g' in self.model_type:
            no_decay.add('goal_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
                % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], 
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, timesteps, actions=None, goals=None, mask=None):
        state_embeddings = self.state_encoder(states)
        batch_size = states.shape[0]
        shape = (batch_size, self.block_size, self.config.n_embd)
        token_embeddings = torch.zeros(
            shape, dtype=torch.float32, device=state_embeddings.device)
        if self.model_type in ['s', 's+g', 's+g+p']:
            token_embeddings[:,:state_embeddings.shape[1],:] = state_embeddings
        if self.model_type in ['s+a', 's+a+g', 's+a+g+p']:
            token_embeddings[:,:state_embeddings.shape[1]*2:2,:] = state_embeddings 
            if actions is not None:
                action_embeddings = self.action_encoder(actions)
                if action_embeddings.shape[1] == state_embeddings.shape[1] - 1: 
                    token_embeddings[  # Eval mode.
                        :,1:state_embeddings.shape[1]*2-1:2,:] = action_embeddings          
                else:
                    token_embeddings[
                        :,1:state_embeddings.shape[1]*2:2,:] = action_embeddings          

        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        timesteps_r = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        position_embeddings = torch.gather(
            all_global_pos_emb, 1, timesteps_r.long())  # BS x 1 x D
        if self.model_type in ['s', 's+g', 's+g+p']:
            position_embeddings = position_embeddings + self.pos_emb
        if self.model_type in ['s+a', 's+a+g', 's+a+g+p']:
            local_pos_emb = torch.repeat_interleave(self.pos_emb, 2, dim=1)
            position_embeddings = position_embeddings + local_pos_emb
        x = token_embeddings + position_embeddings
       
        goals_gt = goals
        if '+p' in self.model_type:
            if self.variant in ['', 'v2', 'v4']:
                goals = self.goal_pred(states[:, 0])
                goals = torch.split(goals, self.state_dim, dim=-1)
                goals = torch.stack(goals, 1)

            # goals_1 = self.goal_pred1(states[:, 0])
            # goals_2 = self.goal_pred2(states[:, 0])
            # goals = torch.stack([goals_1, goals_2], 1)
        if '+g' in self.model_type:
            ########
            # goals = goals_gt[:, -3:]
            # goals = torch.cat([goals_gt[:, :-1], goals[:, -1:]], 1)
            if self.variant == '':
                goal_embeddings = self.state_encoder(goals)
            elif self.variant == 'v2':
                goal_embeddings = self.state_encoder(goals.detach())
            elif self.variant == 'v3':
                goal_embeddings = self.state_encoder(goals_gt)
            elif self.variant == 'v4':
                goal_embeddings = self.state_encoder_v2(goals)
            else:
                goal_embeddings = None
            
            if goal_embeddings is not None:
                goal_embeddings = goal_embeddings + self.goal_pos_emb 
            else:
                goal_embeddings = self.goal_pos_emb.repeat(x.size(0), 1, 1)
            x = torch.cat([goal_embeddings, x], 1)
        
        x = self.drop(x)
        x, outs = self.blocks_wrapper(x, mask=mask)
        x = self.ln_f(x)
        preds = self.action_head(x)

        if self.variant in ['v8', 'v9']:
            next_states_pred = self.state_decoder(outs[-1][:, :-2:2]) ##### -1 -> 0
        elif self.variant in ['v10', 'v11']:
            next_states_pred = self.state_decoder(outs[-1][:, 1:-1:2])
        else:
            next_states_pred = None

        if self.variant in ['v5', 'v6', 'v7']:
            assert self.goal_loss_type
            goals = []
            goal_loss_type = [int(c) for c in self.goal_loss_type]
            if len(goal_loss_type) > 1:
                for l_t in goal_loss_type:
                    goals.append(self.state_decoder_list[l_t](
                        outs[l_t][:,:len(self.goal_type)]))
            else:
                goals.append(self.state_decoder(
                    outs[goal_loss_type[0]][:,:len(self.goal_type)]))

        if '+g' in self.model_type:
            preds = torch.split(
                preds, [len(self.goal_type), token_embeddings.shape[1]], dim=1)[1]
        if self.model_type in ['s', 's+g', 's+g+p']:
            preds = preds[:,:state_embeddings.shape[1]]
        if self.model_type in ['s+a', 's+a+g', 's+a+g+p']:
            preds = preds[:,:state_embeddings.shape[1]*2:2]

        pred_progress = None
        if self.progress_monitor > 0:
            if '+g' in self.model_type:
                last_out = torch.split(
                    outs[-1], [len(self.goal_type), token_embeddings.shape[1]], dim=1)[1]
            else:
                last_out = outs[-1]
            pred_progress = self.progress_classifier(last_out[:, ::2])

        return (
            preds, 
            goals if isinstance(goals, list) else [goals], 
            next_states_pred,
            pred_progress,
        )

    def forward_goal_only(self, states):
        # # V1
        goals = self.goal_pred(states[:, 0])
        goals = torch.split(goals, self.state_dim, dim=-1)
        goals = torch.stack(goals, 1)
        # V2
        # goals_1 = self.goal_pred1(states[:, 0])
        # goals_2 = self.goal_pred2(states[:, 0])
        # goals = torch.stack([goals_1, goals_2], 1)
        return goals


# from data import MS2_MP_Traj_Dataset

# dataset = MS2_MP_Traj_Dataset(
#     control_mode='pd_joint_delta_pos', length=10, seed=0,
#     min_traj_length=50, max_traj_length=BLOCK_SIZE)
#     # rew_thresh='<0.5', rew_filter_delay=1)
#     # override_traj_path=args.override_traj_path)

# # print(len(dataset))
# print(dataset.max_steps)
# d = next(iter(data_loader))
# # d = next(iter(dataset))
# print(d['s'].shape, d['a'].shape, d['timesteps'].shape, d['lengths'])
# # exit()
