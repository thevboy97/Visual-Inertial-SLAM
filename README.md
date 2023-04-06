# Visual-Inertial-SLAM
Implementation of Visual-Inertial (VI) SLAM for an autonomous car in Python using synchronized sensor measurements from an Inertial Measurement Unit (IMU) and a stereo camera on the car. Exercised the Extended Kalman Filter (EKF) for IMU pose prediction to get car trajectory over time and for landmark map update to get landmark locations as observed by the car over time.

Note:

1) Slight changes made to pr3_utils in the visualization function.
2) Part (a) plots taken in dead_reckon.py
3) Part (b) plots taken in feat_mapping.py
4) Part (c), i.e., full VI-SLAM implementaion done in main.py
5) Need scipy module to implement expm() functionality.
