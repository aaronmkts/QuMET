from gym.envs.registration import register

register(id='QuantumArchSearch-v0',
         entry_point='gyms.environments:QuantumArchSearch',
         nondeterministic=True)

register(id='BasicTwoQubit-v0',
         entry_point='gyms.environments:BasicTwoQubitEnv',
         nondeterministic=True)

register(id='BasicThreeQubit-v0',
         entry_point='gyms.environments:BasicThreeQubitEnv',
         nondeterministic=True)

register(id='BasicNQubit-v0',
         entry_point='gyms.environments:BasicNQubitEnv',
         nondeterministic=True)

register(id='NoisyTwoQubit-v0',
         entry_point='gyms.environments:NoisyTwoQubitEnv',
         nondeterministic=True)

register(id='NoisyThreeQubit-v0',
         entry_point='gyms.environments:NoisyThreeQubitEnv',
         nondeterministic=True)

register(id='NoisyNQubit-v0',
         entry_point='gyms.environments:NoisyNQubitEnv',
         nondeterministic=True)