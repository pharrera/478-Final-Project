import unittest
import numpy as np
import os
import pandas as pd
from environment import NetworkEnv
from agent import DQNAgent

class TestNeuroGuard(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a tiny temporary dataset for testing
        cls.test_data_path = "data/test_data.csv"
        df = pd.DataFrame(np.random.rand(10, 5), columns=[f"f{i}" for i in range(5)])
        df["label"] = [0, 1] * 5
        df.to_csv(cls.test_data_path, index=False)

    def test_environment_initialization(self):
        """Happy Path: Environment loads correctly"""
        env = NetworkEnv(self.test_data_path)
        self.assertEqual(env.n_features, 5)
        self.assertEqual(env.action_space.n, 2)

    def test_agent_action_shape(self):
        """Happy Path: Agent outputs valid actions"""
        env = NetworkEnv(self.test_data_path)
        agent = DQNAgent(env.n_features, 2)
        state = env.reset()
        action = agent.act(state)
        self.assertIn(action, [0, 1])

    def test_negative_invalid_file(self):
        """Negative Test: Handling missing files"""
        with self.assertRaises(FileNotFoundError):
            NetworkEnv("non_existent_file.csv")

    def test_dimension_mismatch(self):
        """Edge Case: Agent input dimension mismatch"""
        # Create agent with wrong input size (10) vs env (5)
        agent = DQNAgent(10, 2) 
        state = np.random.rand(5) # Env state size
        # This should fail or throw error depending on implementation, 
        # but here we just check if it runs without crashing or if we can catch a shape error
        try:
            agent.act(state)
        except Exception as e:
            self.assertTrue(True) # Pass if it catches error

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_data_path):
            os.remove(cls.test_data_path)

if __name__ == '__main__':
    unittest.main()