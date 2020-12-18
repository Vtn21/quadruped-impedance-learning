"""spot_controller controller."""

from deepbots.robots.controllers.robot_emitter_receiver_csv \
    import RobotEmitterReceiverCSV
import numpy as np


class SpotRobot(RobotEmitterReceiverCSV):

    def __init__(self):
        super().__init__()
        # Get all 12 motors and position sensors
        self.motors = []
        self.pos_sensors = []
        self.motorNames = ['rear right shoulder abduction motor',
                           'rear right shoulder rotation motor',
                           'rear right elbow motor',
                           'rear left shoulder abduction motor',
                           'rear left shoulder rotation motor',
                           'rear left elbow motor',
                           'front right shoulder abduction motor',
                           'front right shoulder rotation motor',
                           'front right elbow motor',
                           'front left shoulder abduction motor',
                           'front left shoulder rotation motor',
                           'front left elbow motor']
        for i, name in enumerate(self.motorNames):
            self.motors.append(self.robot.getMotor(name))
            self.pos_sensors.append(self.motors[i].getPositionSensor())
            self.pos_sensors[i].enable(self.get_timestep())
        self.last_pos = np.zeros(12, dtype=np.float32)

    def create_message(self):
        # Get all 12 position sensor values
        pos = np.array([sensor.getValue() for sensor in self.pos_sensors],
                       dtype=np.float32)
        # Compute instantaneous velocity
        vel = (pos - self.last_pos)*1000/self.get_timestep()
        # Update last position
        self.last_pos = pos
        # Concatenate arrays
        message = np.concatenate((pos, vel), axis=None)
        # Convert to string and return
        return [str(num) for num in message]

    def use_message_data(self, message):
        # Receive torque values from the supervisor and apply to the motors
        for i, motor in enumerate(self.motors):
            motor.setTorque(float(message[i]))


# Create the robot controller and run it
robot_controller = SpotRobot()
robot_controller.run()
