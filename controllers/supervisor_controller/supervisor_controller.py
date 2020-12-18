from deepbots.supervisor.controllers.supervisor_emitter_receiver \
    import SupervisorCSV

from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec

from tf_agents.trajectories import time_step as ts

import numpy as np

import pinocchio as pin
import tsid

import os

import time

DISCOUNT = np.float32(0.9)

CONTACT_NORMAL = np.array([0.0, 0.0, 1.0])
MU = 0.3
F_MIN = 1.0
F_MAX = 400.0
KP_CONTACT = 10.0


class SpotSupervisor(SupervisorCSV):

    def __init__(self):
        super().__init__()
        self.robot = None
        self.respawnRobot()
        self.message_received = None
        self.message = np.zeros(24, dtype=np.float32)

    def respawnRobot(self):
        """Despawns the existing robot and respawns it in initial position.
        """
        if self.robot is not None:
            # Despawn existing robot
            self.robot.remove()
        # Get the root node of the scene tree
        root_node = self.supervisor.getRoot()
        # Get a list of all children (objects) of the scene
        children_field = root_node.getField("children")
        # Load robot from file and add to last position
        children_field.importMFNode(-1,
            os.path.dirname(os.path.abspath(__file__)) + "/SpotRobot.wbo")
        # Get new robot reference
        self.robot = self.supervisor.getFromDef("SPOT")

    def get_contacts(self):
        """Show which feet are in contact with the ground.

        Uses a numpy float array, with 1 where there is contact, 0 otherwise.

        Order: rear right, rear left, front right, front left.

        Returns:
            Numpy array: [rear right, rear left, front right, front left]
        """
        feet = ["REAR_RIGHT_FOOT", "REAR_LEFT_FOOT",
                "FRONT_RIGHT_FOOT", "FRONT_LEFT_FOOT"]
        contacts = np.zeros(4, dtype=np.float32)
        for i, foot in enumerate(feet):
            if self.supervisor.getFromDef(foot).getNumberOfContactPoints() > 0:
                contacts[i] = 1.0
        return contacts

    def get_observations(self):
        """Get all observations from the simulation.

        - 6 robot velocities (linear + angular)
        - 12 joint angles
        - 12 joint velocities
        - 4 foot contacts

        Returns:
            NumPy array: velocities + angles + contacts
        """
        # Robot velocities (linear + angular)
        vel = np.array(self.robot.getVelocity(), dtype=np.float32)
        # Joint angles
        message_received = self.handle_receiver()
        # Update angles when a message is received
        if message_received is not None:
            self.message_received = message_received
            self.message = np.array(self.message_received, dtype=np.float32)
        # Return the concatenated array
        return np.concatenate((vel, self.message, self.get_contacts()),
                              axis=None)

    def get_reward(self, action):
        # Compute instantaneous power consumption
        power = np.multiply(self.get_observations()[18:30], action)
        total_power = np.sum(np.abs(power))
        return total_power

    def is_done(self):
        """Check if the current episode can be finished.

        Returns:
            Bool: True if episode can be finished, False otherwise
        """
        # Retrieve robot position
        pos = self.robot.getPosition()
        # Check if robot has moved sideways too much
        if abs(pos[0]) > 2.0:
            return True
        # Check if robot has fallen (body too close to the ground)
        elif pos[1] < 0.3:
            return True
        # Check it the robot has reached the end of the track
        elif pos[2] < -20.0:
            return True
        # Check if the robot has walked backwards
        elif pos[2] > 25.0:
            return True
        # No conditions reached, not done yet
        else:
            return False

    def reset(self):
        """Resets the simulation.

        Returns:
            Numpy array: initial observation (all zeros)
        """
        self.respawnRobot()
        self.supervisor.simulationResetPhysics()
        self.message_received = None
        return np.zeros(22, dtype=np.float32)

    def get_info(self):
        """Used for debugging, no practical use here.

        Returns:
            None
        """
        return None


class SpotEnvironment(PyEnvironment):

    def __init__(self, spot_supervisor: SpotSupervisor):
        # Initialize superclass
        super().__init__()
        # Grab the deepbots supervisor
        self.spot_supervisor = spot_supervisor
        # Create the TSID robot wrapper
        path = "../../urdf"
        urdf = path + "/spot.urdf"
        vec = pin.StdVec_StdString()
        vec.extend(item for item in path)
        self.robot = tsid.RobotWrapper(urdf, vec, pin.JointModelFreeFlyer(),
                                       False)
        # Create inverse dynamics formulation
        self.inv_dyn = tsid.InverseDynamicsFormulationAccForce("tsid",
                                                               self.robot,
                                                               False)
        self.q = np.zeros(self.robot.np)
        self.v = np.zeros(self.robot.nv)
        self.inv_dyn.computeProblemData(0.0, self.q, self.v)
        self.data = self.inv_dyn.data()
        # Create a posture task (motion)
        self.posture_task = tsid.TaskJointPosture("task-posture", self.robot)
        self.inv_dyn.addMotionTask(self.posture_task, 1e-3, 1.0, 0.0)
        # Create the solver
        self.solver = tsid.SolverHQuadProgFast("qp solver")
        # Create an array to store the contact points
        # Rear right, rear left, front right, front left
        self.contacts = [False, False, False, False]

    def update_contacts(self):
        frames = ["rear_right_contact", "rear_left_contact",
                  "front_right_contact", "front_left_contact"]
        new_contacts = np.array(self.spot_supervisor.get_contacts(),
                                dtype=np.bool)
        for i in len(self.contacts):
            if self.contacts[i] != new_contacts[i]:
                if new_contacts[i]:
                    # Configure a new contact
                    contact = tsid.ContactPoint(frames[i], self.robot,
                                                frames[i], CONTACT_NORMAL,
                                                MU, F_MIN, F_MAX)
                    contact.setKp(KP_CONTACT * np.ones(3))
                    contact.setKd(2.0 * np.sqrt(KP_CONTACT) * np.ones(3))
                    frame_id = self.robot.model().getFrameId(frames[i])
                    frame_ref = self.robot.framePosition(self.data, frame_id)
                    contact.setReference(frame_ref)
                    contact.useLocalFrame(False)
                    # Add new contact
                    self.inv_dyn.addRigidContact(contact, 1e-5, 1.0, 1.0)
                else:
                    # Remove this contact
                    self.inv_dyn.removeRigidContact(frames[i], 0.5)
                self.contacts[i] = new_contacts[i]

    def observation_spec(self):
        """Observation spec for the Spot environment.

        Returns:
            ArraySpec: a float32 array with 23 values:
            - 1 reference velocity (linear)
            - 6 robot velocities (linear + angular)
            - 12 joint angles
            - 4 foot contacts
        """
        return ArraySpec(shape=(23,), dtype=np.float32)

    def action_spec(self):
        """Action spec for the Spot environment.

        Kp: proportional gain (stiffness). Must be positive.
        Kd: derivative gain (damping). Must be positive.
        Reference position (radians): range defined per joint.

        Returns:
            BoundedArraySpec: a 3 x 12 float32 matrix, containing the values of
            Kp, Kd and reference position for each motor.
        """
        min_Kp_Kd = np.zeros(12)
        max_Kp_Kd = np.ones(12)*np.inf
        min_pos = np.array([-0.6, -1.7, -0.45, -0.6, -1.7, -0.45,
                            -0.6, -1.7, -0.45, -0.6, -1.7, -0.45])
        max_pos = np.array([0.5, 1.7, 1.6, 0.5, 1.7, 1.6,
                            0.5, 1.7, 1.6, 0.5, 1.7, 1.6])
        return BoundedArraySpec(shape=(3, 12), dtype=np.float32,
                                minimum=[min_Kp_Kd, min_Kp_Kd, min_pos],
                                maximum=[max_Kp_Kd, max_Kp_Kd, max_pos],
                                name="action")

    def _reset(self):
        """Resets the environment, also resetting the simulation.

        Returns:
            TimeStep: initial time step, with zero reward, default discount
            rate and initial observation (all zeros).
        """
        self.spot_supervisor.reset()
        return ts.TimeStep(ts.StepType.FIRST, np.float32(0.0), DISCOUNT,
                           np.zeros(23, dtype=np.float32))

    def _step(self, action):
        # Parse the action array
        Kp = action[0]
        Kd = action[1]
        ref_pos = action[2]
        # Set the gains of the posture task
        self.posture_task.setKp(Kp)
        self.posture_task.setKd(Kd)
        # Set the desired trajectory
        traj_posture = tsid.TrajectoryEuclidianConstant("traj_joint", ref_pos)
        self.posture_task.setReference(traj_posture.computeNext())
        # Update contact definition
        self.update_contacts()
        # Update size of solver
        self.solver.resize(self.inv_dyn.nVar,
                           self.inv_dyn.nEq,
                           self.inv_dyn.nIn)
        # Compute problem data
        hqp_data = self.inv_dyn.computeProblemData(t, self.q, self.v)
        sol = self.solver.solve(hqp_data)
        # Get desired torque
        tau = self.inv_dyn.getActuatorForces(sol)
        # Send torque to actuators through the supervisor
        self.spot_supervisor.step(tau)


if __name__ == "__main__":
    spot_supervisor = SpotSupervisor()
    action = np.ones(12)
    for _ in range(50):
        spot_supervisor.step(action)
        print(spot_supervisor.get_reward(action))
