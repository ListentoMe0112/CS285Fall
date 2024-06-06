from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        batch_size = observations.shape[0]
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            next_qa_values = self.target_critic(next_observations)
            # Use the actor to compute a critic backup
            next_action_dist:torch.distributions.Distribution = self.actor(next_observations)
            # values = torch.sum(torch.exp(action_dist.logits) * next_qa_values, dim = 1)
            next_values = torch.sum(torch.exp(next_action_dist.logits) * next_qa_values, dim = 1)
            assert next_values.shape == rewards.shape == (batch_size, ), "next_qs.shape: {}, rewards.shape: {}".format(next_values.shape, rewards.shape)
            # TODO(student): Compute the TD target
            target_values = rewards + self.discount * next_values * (1 - dones.int())

        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = torch.squeeze(torch.gather(qa_values, 1, torch.unsqueeze(actions, dim = 1)), dim = 1)
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        batch_size = observations.shape[0]
        with torch.no_grad():
            qa_values = self.target_critic(observations)
            q_values = torch.squeeze(torch.gather(qa_values, 1, torch.unsqueeze(actions, dim = 1)), dim = 1)
            
            assert action_dist.logits.shape == qa_values.shape == (action_dist.logits * qa_values).shape == (batch_size, self.num_actions),"action_dist.logits.shape: {}, qa_values.shape: {}, multi_shape: {}".format(action_dist.logits.shape, qa_values.shape, (action_dist.logits * qa_values).shape)
            # values = torch.sum(torch.exp(action_dist.logits) * qa_values, dim = 1)
            values = torch.sum(torch.exp(action_dist.logits) * qa_values, dim = 1)

            assert q_values.shape == values.shape == (batch_size,),  "q_values.shape: {}, values.shape: {}".format(q_values.shape, values.shape)
            advantages = q_values - values
            return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        batch_size = observations.shape[0]
        action_dist = self.actor(observations)

        assert action_dist.log_prob(actions).shape == (batch_size,), "action_dist.shape: {}, batch_size: {}".format(action_dist.log_prob(actions).shape, batch_size)

        loss = -torch.mean(action_dist.log_prob(actions) * torch.exp( 1/self.temperature * self.compute_advantage(observations, actions, action_dist)))

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
