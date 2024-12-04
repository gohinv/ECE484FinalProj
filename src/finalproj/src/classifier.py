# scripts/stop_sign_detector.py
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image



### NOT TESTED

class StopSignDetector:
    def __init__(self):
        self.bridge = CvBridge()
        cascade_path = '../cascade/stop_sign_classifier.xml'
        self.stop_sign_cascade = cv2.CascadeClassifier(cascade_path)
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/camera/processed_image', Image, queue_size=10) # we want to publish len(stop_signs)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            stop_signs = self.stop_sign_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in stop_signs:
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            processed_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(processed_image)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    rospy.init_node('stop_sign_detector', anonymous=True)


    # creates the model and continues running; callbacks will allow for functionality.
    detector = StopSignDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
