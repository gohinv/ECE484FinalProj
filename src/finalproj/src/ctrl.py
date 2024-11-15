
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

class PIDController:
    def __init__(self):
        rospy.init_node('pid_controller', anonymous=True)
        
        # PID parameters
        self.kp = 0.5  # Proportional gain
        self.ki = 0.0  # Integral gain
        self.kd = 0.1  # Derivative gain
        
        # PID variables
        self.prev_error = 0.0
        self.integral = 0.0
        
        self.max_steering_angle = 0.34  # Maximum steering angle (in radians)
        self.desired_speed = 1.0        # Desired speed (in meters per second)
        
        # Time variables for derivative calculation
        self.prev_time = rospy.Time.now()
        
        # Subscribe to the lane detection error
        rospy.Subscriber('lane_detection/error', Float32, self.error_callback)
        
        # Publisher for the drive commands
        self.drive_pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
        
        # Current error
        self.current_error = 0.0
        
        # Main loop rate
        self.rate = rospy.Rate(20)  # 20 Hz
    
    def error_callback(self, msg):
        # Update the current error from the message
        self.current_error = msg.data
        
    def run(self):
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            delta_time = (current_time - self.prev_time).to_sec()
            
            # Protect against division by zero
            if delta_time == 0:
                delta_time = 0.0001
            
            # PID calculations
            error = self.current_error
            self.integral += error * delta_time
            derivative = (error - self.prev_error) / delta_time
            
            steering_angle = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
            
            # Clamp the steering angle to the maximum allowed value
            steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))
            
            # Create and publish the drive message
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = current_time
            drive_msg.header.frame_id = "base_link"
            drive_msg.drive.steering_angle = steering_angle
            drive_msg.drive.speed = self.desired_speed
            
            self.drive_pub.publish(drive_msg)
            
            # Update previous values for next iteration
            self.prev_error = error
            self.prev_time = current_time
            
            # Sleep for the remainder of the loop
            self.rate.sleep()
            
if __name__ == '__main__':
    try:
        controller = PIDController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
