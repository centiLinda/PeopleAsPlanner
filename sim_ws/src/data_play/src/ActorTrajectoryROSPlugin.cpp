#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>

namespace gazebo
{
  // GZ_REGISTER_MODEL_PLUGIN(ActorTrajectoryROSPlugin);

  #define WALKING_ANIMATION "walking"

  class ActorTrajectoryROSPlugin : public ModelPlugin
  {
  private:
    ros::NodeHandle* nh;
    ros::Publisher pub;
    // physics::ActorPtr actor;
    physics::ModelPtr model;
    std::vector<event::ConnectionPtr> connections;
    std::string trajectory_file;
    std::vector<std::vector<double>> trajectory;
    size_t currentIndex = 0;
    double animationFactor = 1.0;
    ros::Subscriber intSubscriber;
    int receivedInt = 0;
    double dt=1.0/30;
    double begin_time_sec=0;
    int path_index=0;
    std::vector<double> waiting_position;

    physics::TrajectoryInfoPtr trajectoryInfo;

    float ten_hz_time=0.1;
    float last_time=0;

  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    { 
      if (!ros::isInitialized()) {
          ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized."
                           " Make sure you call ros::init() in the main thread.");
          return;
      }
      this->model=_model;
      this->nh = new ros::NodeHandle("~");

      // // Get the actor from the model
      // this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);

      // if (!this->actor)
      // {
      //   ROS_ERROR("This plugin must be attached to an Actor model.");
      //   return;
      // }

      if (_sdf->HasElement("trajectory")) {
          this->trajectory_file = _sdf->Get<std::string>("trajectory");
      } else {
          ROS_ERROR("Trajectory file not found. Please check the world config.");
          return;
      }

    if (_sdf->HasElement("wait_position")) {
        sdf::ElementPtr waitElem = _sdf->GetElement("wait_position");

        while (waitElem) {
            ignition::math::Vector3d pos = waitElem->Get<ignition::math::Vector3d>(); // Extract 3D position
            this->waiting_position.push_back(pos.X());
            this->waiting_position.push_back(pos.Y());
            this->waiting_position.push_back(pos.Z());

            ROS_INFO_STREAM("Loaded wait_position: " << pos.X() << ", " << pos.Y() << ", " << pos.Z());

            waitElem = waitElem->GetNextElement("wait_position");  // Move to the next <wait_position> element
        }
    } else {
          ROS_ERROR("Wait_position not found. Please check the world config.");
          return;
    }
    LoadTrajectory(this->trajectory_file);

    // auto skelAnims = this->actor->SkeletonAnimations();
    // if (skelAnims.find(WALKING_ANIMATION) == skelAnims.end()) {
    //   ROS_ERROR_STREAM("Skeleton animation " << WALKING_ANIMATION << " not found.");
    // } else {
    //   // Create custom trajectory
    //   this->trajectoryInfo.reset(new physics::TrajectoryInfo());
    //   this->trajectoryInfo->type = WALKING_ANIMATION;
    //   this->trajectoryInfo->duration = 1.0;

    //   this->actor->SetCustomTrajectory(this->trajectoryInfo);
    // }
    
    this->connections.push_back(event::Events::ConnectWorldUpdateBegin(std::bind(&ActorTrajectoryROSPlugin::OnUpdate, this, std::placeholders::_1)));
    // Subscribe to an integer topic
    this->intSubscriber = this->nh->subscribe("/env_control", 1, &ActorTrajectoryROSPlugin::IntCallback, this);

    ROS_INFO("ActorTrajectoryROSPlugin successfully loaded.");
    }

    void LoadTrajectory(const std::string &filePath)
    {
      std::ifstream file(filePath);
      if (!file)
      {
        ROS_ERROR_STREAM("Unable to open trajectory file: " << filePath);
        return;
      }

      std::string line;
      while (std::getline(file, line))
      {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value)
        {
          row.push_back(value);
          if (ss.peek() == '\t' || ss.peek() == ' ')
            ss.ignore();
        }
        if (row.size() == 6) // Ensure valid row format
          trajectory.push_back(row);
      }

      file.close();
      ROS_INFO_STREAM("Loaded " << trajectory.size() << " trajectory points.");

      printTrajectory();
    }

    void printTrajectory() {
        ROS_INFO("Printing loaded trajectory:");
        for (size_t i = 0; i < this->trajectory.size(); ++i) {
            std::ostringstream oss;
            oss << "Step " << i << ": ";
            for (size_t j = 0; j < this->trajectory[i].size(); ++j) {
                oss << this->trajectory[i][j] << " ";
            }
            ROS_INFO_STREAM(oss.str());
        }
    }

    void IntCallback(const std_msgs::Int32::ConstPtr &msg)
    {
      this->receivedInt = msg->data; // Store the received integer value
      ROS_INFO_STREAM("Received ROS int: " << receivedInt);
      if(msg->data==0){
        this->begin_time_sec=0;
        this->path_index=0;
      }else if(msg->data==1){
        this->begin_time_sec= ros::Time::now().toSec();
        this->path_index=0;
      }
    }

    void OnUpdate(const common::UpdateInfo &_info)
    {

      if (this->receivedInt == 0) {
          if (!this->waiting_position.empty()) {
              // Ensure we have at least X, Y, and Z values
              if (this->waiting_position.size() >= 3) {
                  ignition::math::Pose3d actorPose = this->model->WorldPose();

                  // Set position to waiting position (start position)
                  actorPose.Pos().X(this->waiting_position[0]);
                  actorPose.Pos().Y(this->waiting_position[1]);
                  actorPose.Pos().Z(this->waiting_position[2]);

                  // Reset rotation to default (0 yaw) to avoid unwanted rotation
                  actorPose.Rot() = ignition::math::Quaterniond(0, 0, 0); 
                  // this->actor->Reset();  // Reset animation to initial state
                  // // Apply the reset position
                  // this->actor->SetWorldPose(actorPose, false, false);
                  
                  // this->actor->SetScriptTime(0.0);
                  this->model->SetWorldPose(actorPose);
                  // this->actor->SetScriptTime(this->actor->ScriptTime());
                  // ROS_INFO_STREAM("Model reset to waiting position: "
                  //                 << this->waiting_position[0] << ", "
                  //                 << this->waiting_position[1] << ", "
                  //                 << this->waiting_position[2]);
              } else {
                  ROS_ERROR("Not enough values in waiting_position to set the start position!");
              }
          } else {
              ROS_ERROR("Waiting position is empty! Cannot reset the model.");
          }

          return; // Stop further updates since the model is in waiting mode
      } else if (this->receivedInt == 1) {
        // if ( ros::Time::now().toSec()-last_time>ten_hz_time){
        //   last_time=ros::Time::now().toSec();
        // }else{
        //   return;
        // }
        // ROS_INFO("I am here");

        if (this->trajectory.empty()) {
          ROS_ERROR("No trajectory loaded!");
          return;
        }
        double currentTime=ros::Time::now().toSec()-this->begin_time_sec;
        if(currentTime<trajectory[0][1]*dt){
          return;
        }
        ROS_INFO("I am here3 | path_index: %f", path_index);
        while((int(trajectory[this->path_index+1][1])*this->dt)<currentTime){
          if(this->path_index==this->trajectory.size()-2){
            // if(currentTime>this->trajectory[this->trajectory.size()-1][1]*this->dt){
            //   this->receivedInt=0;
            //   return;
            // }
            receivedInt=0;
            return;
          }else{
            path_index=path_index+1;
            // ROS_INFO("I am here4 | Current Time: %f", currentTime);
            // ROS_INFO("I am here4 | path_index: %d", path_index);
            // ROS_INFO("I am here4 | frame: %f", trajectory[this->path_index+1][1]);
            // ROS_INFO("I am here4 | frametime: %f", (int(trajectory[this->path_index+1][1])*this->dt));
            // ROS_INFO("I am here4 | dt: %f", dt);
          }
        }
        // ROS_INFO("I am here3");
        // TODO calculate the intersection of the code
        double time_over_last=currentTime-trajectory[path_index][1]*dt;

        double time_duration= trajectory[path_index+1][1]*dt-trajectory[path_index][1];
        double x_last=trajectory[path_index][2];
        double y_last=trajectory[path_index][3];

        double x_future=trajectory[path_index+1][2];
        double y_future=trajectory[path_index+1][3];

        double x_now=x_last+(x_future-x_last)*time_over_last/time_duration;
        double y_now=y_last+(y_future-y_last)*time_over_last/time_duration;


    // Compute yaw (heading direction)
    double dx = x_future - x_last;
    double dy = y_future - y_last;
    double yaw = std::atan2(dy, dx);  // Ensure smooth rotation

    // Set the new actor pose
    ignition::math::Pose3d actorPose;
    actorPose.Pos().X(x_now);
    actorPose.Pos().Y(y_now);
    actorPose.Pos().Z(1.0);  // Maintain ground level
    actorPose.Rot() = ignition::math::Quaterniond(0, 0, yaw);  // Apply correct yaw

    // double distanceTraveled =(actorPose.Pos() - this->actor->WorldPose().Pos()).Length();
    // // Compute distance traveled for animation timing
    // // double distanceTraveled = std::sqrt(dx * dx + dy * dy);
    // // Move the actor
    // this->actor->SetWorldPose(actorPose, false, false);
    // this->actor->SetScriptTime(this->actor->ScriptTime() + (distanceTraveled * this->animationFactor));
    // double distanceTraveled = (actorPose.Pos() - this->actor->WorldPose().Pos()).Length();
    // this->actor->SetWorldPose(actorPose, false, false);
    // this->actor->SetCustomTrajectory(this->trajectoryInfo);

    // this->actor->SetScriptTime(this->actor->ScriptTime() + (distanceTraveled*dt * this->animationFactor));
  this->model->SetWorldPose(actorPose);
  this->model->SetLinearVel(ignition::math::Vector3d(dx, dy, 0));
    ROS_INFO_STREAM("Moving to path index " << path_index 
                    << " | Position: " << x_now << ", " << y_now 
                    << " | Yaw: " << yaw);
    }
        
      // Placeholder function - re-enable this logic when needed
    }
  };

  GZ_REGISTER_MODEL_PLUGIN(ActorTrajectoryROSPlugin)
}
