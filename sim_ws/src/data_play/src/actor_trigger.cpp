#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <fstream>
#include <sstream>

namespace gazebo
{
    class ActorTriggerPlugin: public ModelPlugin
    {
        private:
            ros::NodeHandle* nh;
            ros::Publisher pub;
            physics::ActorPtr actor;
            std::vector<event::ConnectionPtr> connections;
            ros::Subscriber intSubscriber;
            int states=0;
            int receivedInt;
            std::vector<double> waiting_position;
            ignition::math::Pose3d wait_pos;
            ignition::math::Pose3d first_pos;
            std::string trajectory_file;

        public:
            void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
            {
                ROS_ERROR("Loading");

                if (!ros::isInitialized()) {
                    ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized."
                        " Make sure you call ros::init() in the main thread.");
                        return;
                }

                this->nh = new ros::NodeHandle("~");
                // Get the actor from the model
                this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
                if (!this->actor)
                {
                    ROS_ERROR("This plugin must be attached to an Actor model.");
                    return;
                }
                this->intSubscriber = this->nh->subscribe("/env_control", 10, &ActorTriggerPlugin::IntCallback, this);
                // ROS_INFO("ActorTriggerRosPlugin successfully loaded.");
                this->connections.push_back(event::Events::ConnectWorldUpdateBegin(std::bind(&ActorTriggerPlugin::OnUpdate, this, std::placeholders::_1)));
            }

            void IntCallback(const std_msgs::Int32::ConstPtr &msg)
            {
                this->receivedInt = msg->data; // Store the received integer value
                ROS_INFO_STREAM("Received ROS int: " << receivedInt);
                if(msg->data==1){
                    this->actor->Play();
                }else{
                    this->actor->Stop();
                }
            }
            void OnUpdate(const common::UpdateInfo &_info)
            {
                if(this->actor->IsActive()){
                //    ROS_INFO_STREAM("Running: info size:"<<this->actor->trajInfo.size()); 
                }else{

                }
            }


    };
    GZ_REGISTER_MODEL_PLUGIN(ActorTriggerPlugin)

}