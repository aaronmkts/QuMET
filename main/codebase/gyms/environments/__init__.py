from gym.envs.registration import register

register(id='QuantumArchSearch-v0',
         entry_point='environments:QuantumArchSearch',
         nondeterministic=True)

register(id='BasicTwoQubit-v0',
         entry_point='environments:BasicTwoQubitEnv',
         nondeterministic=True)

register(id='BasicThreeQubit-v0',
         entry_point='environments:BasicThreeQubitEnv',
         nondeterministic=True)

register(id='BasicNQubit-v0',
         entry_point='environments:BasicNQubitEnv',
         nondeterministic=True)

register(id='NoisyTwoQubit-v0',
         entry_point='environments:NoisyTwoQubitEnv',
         nondeterministic=True)

register(id='NoisyThreeQubit-v0',
         entry_point='environments:NoisyThreeQubitEnv',
         nondeterministic=True)

register(id='NoisyNQubit-v0',
         entry_point='environments:NoisyNQubitEnv',
         nondeterministic=True)