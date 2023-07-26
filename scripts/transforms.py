import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import CameraInfo

from config import BASE_FRAME, CAMERA_FRAME, CAMERA_INFO, DEPTH_CAMERA_INFO


class TfBuffer():
    '''
    Helper class for storing the tf buffer

    This class needs to be initialized before looking up any transforms in the TF tree below
    '''
    @classmethod
    def __init__(cls):
        cls.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(cls.tf_buffer, queue_size=10)

    @classmethod
    def get_tf_buffer(cls):
        return cls.tf_buffer


class Transformer():
    '''
    Helper class for storing the camera to base transformations at a specfic time
    '''
    def __init__(self, tf_buffer: tf2_ros.Buffer):
        # Get the transformation from camera to world
        cam_to_base = tf_buffer.lookup_transform(BASE_FRAME, CAMERA_FRAME, rospy.Time(0), rospy.Duration.from_sec(0.5)).transform
        base_to_cam = tf_buffer.lookup_transform(CAMERA_FRAME, BASE_FRAME, rospy.Time(0), rospy.Duration.from_sec(0.5)).transform

        # Convert the Transform msg to a Pose msg
        pose = Pose(position=Point(
                        x=cam_to_base.translation.x, y=cam_to_base.translation.y, z=cam_to_base.translation.z),
                    orientation=cam_to_base.rotation)

        # NOTE: Not sure why, but the z coordinate needs to be flipped
        pose.position.z *= -1

        self.E_cam_to_world = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        pose = Pose(position=Point(
                        x=base_to_cam.translation.x, y=base_to_cam.translation.y, z=base_to_cam.translation.z),
                    orientation=base_to_cam.rotation)

        self.E_world_to_cam = tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

        # Get the camera intrinsics
        camera_info = rospy.wait_for_message(CAMERA_INFO, CameraInfo, timeout=10)
        depth_info = rospy.wait_for_message(DEPTH_CAMERA_INFO, CameraInfo, timeout=10)

        if camera_info is None or depth_info is None:
            raise RuntimeError(f'Failed to get camera and depth camera info on topics {camera_info} and {depth_info}')

        self.intrinsic = np.array(camera_info.K).reshape((3, 3))
        self.depth_intrinsic = np.array(depth_info.K).reshape((3, 3))

        self.width, self.height = camera_info.width, camera_info.height

    def transform_cam_to_base(self, pt: Point) -> Point:
        '''
        Transform a point from the camera frame to the robot frame for this transform

        Parameters
            pt (geometry_msgs.msg.Point): The point to transform

        Returns
            geometry_msgs.msg.Point: The transformed point
        '''
        # Normalize the point
        x = (pt[0] - self.intrinsic[0, 2]) / self.intrinsic[0, 0]
        y = (pt[1] - self.intrinsic[1, 2]) / self.intrinsic[1, 1]

        # Scale with depth
        x *= pt[2] * 0.001
        y *= pt[2] * 0.001

        # Transform
        transformed_pt = np.matmul(self.E_cam_to_world, np.array([pt[2], x, y, 1]))

        return [transformed_pt[0], transformed_pt[1], transformed_pt[2]]

    def transform_cam_to_base_arr(self, pts: np.ndarray) -> np.ndarray:
        '''
        Transform a point from the camera frame to the robot frame for this transform

        Parameters
            pts (np.ndarray): The points to transform

        Returns
            np.ndarray: The transformed points
        '''
        # Normalize the point
        x = (pts[:, 0] - self.intrinsic[0, 2]) / self.intrinsic[0, 0]
        y = (pts[:, 1] - self.intrinsic[1, 2]) / self.intrinsic[1, 1]

        # Scale with depth
        x *= pts[:, 2] 
        y *= pts[:, 2] 

        # Transform
        transformed_pts = np.matmul(self.E_cam_to_world, np.vstack((pts[:, 2], x, y, np.ones(pts.shape[0]))))
        transformed_pts = transformed_pts.T

        return transformed_pts[:, :3]

    def transform_base_to_cam_pos(self, pt: Point) -> Point:
        '''
        Transform a point from the robot frame to the camera frame for this transform

        Parameters
            pt (geometry_msgs.msg.Point): The point to transform

        Returns
            geometry_msgs.msg.Point: The transformed point
        '''
        # Transform
        transformed_pt = np.matmul(self.E_world_to_cam, np.array([pt.x, pt.y, pt.z, 1]))

        return Point(x=transformed_pt[0], y=transformed_pt[1], z=transformed_pt[2])
