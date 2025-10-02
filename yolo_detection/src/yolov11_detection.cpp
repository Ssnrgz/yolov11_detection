#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "tools/YOLO11.hpp"
#include <mutex>
#include <fstream>
#include <set>
#include <algorithm>

class SignDetectorNode {
public:
    SignDetectorNode(ros::NodeHandle& nh) : nh_(nh), message_count_(0) {
        loadParameters();

        ROS_INFO_STREAM("Using image topic: " << image_topic_);

        trained_classes_ = loadClasses(labels_path_);
        allowed_classes_.insert(trained_classes_.begin(), trained_classes_.end());

        detector_ = std::make_shared<YOLO11Detector>(model_path_, labels_path_, use_gpu_);

        image_sub_ = nh_.subscribe(image_topic_, 10, &SignDetectorNode::imageCallback, this);

        if (enable_debug_window_) {
            cv::namedWindow("YOLO11 Filtered Detections", cv::WINDOW_AUTOSIZE);
        }

        timer_ = nh_.createTimer(ros::Duration(5.0), &SignDetectorNode::timerCallback, this);

        checkTopics();
        ROS_INFO("SignDetectorNode initialized.");
    }

    ~SignDetectorNode() {
        if (enable_debug_window_) {
            cv::destroyAllWindows();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Timer timer_;
    std::shared_ptr<YOLO11Detector> detector_;
    std::vector<std::string> trained_classes_;
    std::set<std::string> allowed_classes_;
    std::mutex mutex_;
    int message_count_;
    ros::Time last_frame_time_;

    std::string image_topic_, model_path_, labels_path_;
    bool use_gpu_, enable_debug_window_;
    float min_confidence_;

    void loadParameters() {
        nh_.param<std::string>("image_topic", image_topic_, "kamera topiğinizi yazınız");
        nh_.param<std::string>("model_path", model_path_, "paketi koyduğunuz dosya yolunu yazınız.../yolo_detection/models/.onnx");
        nh_.param<std::string>("labels_path", labels_path_, "paketi koyduğunuz dosya yolunu yazınız...yolo_detection/models/.txt");
        nh_.param<bool>("use_gpu", use_gpu_, true);
        nh_.param<bool>("enable_debug_window", enable_debug_window_, true);
        nh_.param<float>("min_confidence", min_confidence_, 0.5f);
    }

    std::vector<std::string> loadClasses(const std::string& path) {
        std::ifstream file(path);
        std::vector<std::string> classes;
        std::string line;
        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (!line.empty()) classes.push_back(line);
        }
        ROS_INFO("Loaded %zu classes from %s", classes.size(), path.c_str());
        return classes;
    }

    void checkTopics() {
        ros::master::V_TopicInfo master_topics;
        ros::master::getTopics(master_topics);
        bool found = false;
        for (const auto& topic : master_topics) {
            if (topic.name == image_topic_) {
                found = true;
                break;
            }
        }
        if (!found) {
            ROS_WARN("Topic %s not found!", image_topic_.c_str());
        }
    }

    void timerCallback(const ros::TimerEvent&) {
        ROS_INFO("Messages received so far: %d", message_count_);
        if (image_sub_.getNumPublishers() == 0) {
            ROS_WARN("No publishers for topic: %s", image_topic_.c_str());
        }
    }

    void drawBoundingBoxMask(cv::Mat& image, const std::vector<Detection>& detections) {
        static const std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 0, 255)
        };

        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            
            std::string label = "Unknown";
            if (det.classId >= 0 && det.classId < trained_classes_.size()) {
                label = trained_classes_[det.classId];
            }
            label += " " + std::to_string(int(det.conf * 100)) + "%";

            cv::Scalar color = colors[det.classId % colors.size()];

            cv::Point pt1(det.box.x, det.box.y);
            cv::Point pt2(det.box.x + det.box.width, det.box.y + det.box.height);
            
            cv::rectangle(image, pt1, pt2, color, 2);

            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(det.box.y, labelSize.height);
            cv::rectangle(image,
                          cv::Point(det.box.x, top - labelSize.height - baseLine),
                          cv::Point(det.box.x + labelSize.width, top),
                          color,
                          cv::FILLED);

            cv::putText(image, label,
                        cv::Point(det.box.x, top - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1);
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        message_count_++;
        ros::Time current_time = msg->header.stamp;
        double fps = 0.0;

        if (!last_frame_time_.isZero()) {
            ros::Duration diff = current_time - last_frame_time_;
            if (diff.toSec() > 0.0) {
                fps = 1.0 / diff.toSec();
            }
        }
        last_frame_time_ = current_time;

        ROS_INFO("Received image #%d - Size: %dx%d, Encoding: %s --- FPS: %.2f", 
                 message_count_, msg->width, msg->height, msg->encoding.c_str(), fps);

        try {
            cv_bridge::CvImageConstPtr cv_ptr;
            try {
                if (msg->encoding == "rgb8" || msg->encoding == "bgr8" || msg->encoding == "mono8") {
                    cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
                } else {
                    cv_ptr = cv_bridge::toCvShare(msg);
                    ROS_WARN("Unknown encoding: %s. Using default conversion.", msg->encoding.c_str());
                }
            } catch (const cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge error: %s", e.what());
                return;
            }

            if (!cv_ptr || cv_ptr->image.empty()) {
                ROS_ERROR("Empty image after conversion");
                return;
            }

            cv::Mat frame = cv_ptr->image;
            cv::Mat processed_frame;
            switch (frame.channels()) {
                case 1:
                    cv::cvtColor(frame, processed_frame, cv::COLOR_GRAY2BGR);
                    break;
                case 3:
                    if (msg->encoding == "rgb8") {
                        cv::cvtColor(frame, processed_frame, cv::COLOR_RGB2BGR);
                    } else {
                        processed_frame = frame.clone();
                    }
                    break;
                case 4:
                    cv::cvtColor(frame, processed_frame, cv::COLOR_BGRA2BGR);
                    break;
                default:
                    ROS_ERROR("Unsupported number of channels: %d", frame.channels());
                    return;
            }

            cv::Mat resized;
            try {
                cv::resize(processed_frame, resized, cv::Size(720, 480));
            } catch (const cv::Exception& e) {
                ROS_ERROR("Resize failed: %s", e.what());
                return;
            }

            // FPS yazısı (her karede sol üst köşe)
            cv::putText(resized, cv::format("FPS: %.2f", fps),
                        cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 255, 255), 2);

            if (enable_debug_window_) {
                cv::imshow("YOLO11 Filtered Detections", resized);
                cv::waitKey(1);
            }

            try {
                auto detections = detector_->detect(resized, min_confidence_, 0.9f);

                std::vector<Detection> filtered;
                for (const auto& det : detections) {
                    if (det.classId >= 0 && det.classId < trained_classes_.size()) {
                        const std::string& name = trained_classes_[det.classId];
                        if (allowed_classes_.count(name)) {
                            filtered.push_back(det);
                        }
                    }
                }

                if (!filtered.empty()) {
                    drawBoundingBoxMask(resized, filtered);
                    if (enable_debug_window_) {
                        cv::imshow("YOLO11 Filtered Detections", resized);
                        cv::waitKey(1);
                    }
                }
            } catch (const std::exception& e) {
                ROS_ERROR("YOLO detection failed: %s", e.what());
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in imageCallback: %s", e.what());
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sign_detector_node");
    ros::NodeHandle nh("~");

    ROS_INFO("OpenCV version: %s", CV_VERSION);
    try {
        SignDetectorNode node(nh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Fatal error: %s", e.what());
        return 1;
    }
    return 0;
}

    return 0;
}

