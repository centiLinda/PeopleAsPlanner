#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>
#include <fstream>
#include <sstream>
#include <vector>

namespace gazebo
{
  class TrajectoryFollowerPlugin : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
    {
      // Ensure the model is an actor
      this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
      if (!this->actor)
      {
        gzerr << "TrajectoryFollowerPlugin can only be attached to actors.\n";
        return;
      }

      // Get the trajectory file name from SDF
      if (_sdf->HasElement("trajectory_file"))
      {
        this->trajectoryFile = _sdf->Get<std::string>("trajectory_file");
      }
      else
      {
        gzerr << "No trajectory file specified.\n";
        return;
      }

      // Load the trajectory from the file
      if (!LoadTrajectory(this->trajectoryFile))
      {
        gzerr << "Failed to load trajectory from file: " << this->trajectoryFile << "\n";
        return;
      }

      // Initialize the actor's trajectory
      this->trajectoryInfo.reset(new physics::TrajectoryInfo());
      this->trajectoryInfo->type = "walk";
      this->trajectoryInfo->duration = this->trajectory.back().time;

      this->actor->SetCustomTrajectory(this->trajectoryInfo);

      // Connect to the world update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&TrajectoryFollowerPlugin::OnUpdate, this, std::placeholders::_1));
    }

    void OnUpdate(const common::UpdateInfo &_info)
    {
    // Calculate the elapsed time
    double currentTime = (_info.simTime - this->startTime).Double();

    // Find the current segment in the trajectory
    while (currentSegment < trajectory.size() - 1 &&
            currentTime >= trajectory[currentSegment + 1].time)
    {
        ++currentSegment;
    }

    // Interpolate between the current segment and the next
    if (currentSegment < trajectory.size() - 1)
    {
        const auto &wp1 = trajectory[currentSegment];
        const auto &wp2 = trajectory[currentSegment + 1];
        double t = (currentTime - wp1.time) / (wp2.time - wp1.time);

        // Interpolate position
        ignition::math::Vector3d pos = wp1.pose.Pos() * (1 - t) + wp2.pose.Pos() * t;

        // Interpolate rotation using slerp
        ignition::math::Quaterniond rot = ignition::math::Quaterniond::Slerp(t, wp1.pose.Rot(), wp2.pose.Rot());

        // Combine position and rotation into a pose
        ignition::math::Pose3d pose(pos, rot);

        this->actor->SetWorldPose(pose);
    }
    }

  private:
    struct Waypoint
    {
      double time;
      ignition::math::Pose3d pose;
    };

    bool LoadTrajectory(const std::string &filename)
    {
      std::ifstream file(filename);
      if (!file.is_open())
      {
        gzerr << "Unable to open trajectory file: " << filename << "\n";
        return false;
      }

      std::string line;
      while (std::getline(file, line))
      {
        if (line.empty() || line[0] == '#')
          continue; // Skip empty lines and comments

        std::istringstream iss(line);
        int track_id, frame_id;
        double pos_x, pos_y, dx, dy;
        if (!(iss >> track_id >> frame_id >> pos_x >> pos_y >> dx >> dy))
        {
          gzerr << "Invalid line format in trajectory file: " << line << "\n";
          continue;
        }

        double time = frame_id / 30.0; // Convert frame_id to time
        ignition::math::Pose3d pose(pos_x, pos_y, 0, 0, 0, 0); // Assuming z=0 and no rotation
        trajectory.push_back({time, pose});
      }

      return true;
    }

    physics::ActorPtr actor;
    physics::TrajectoryInfoPtr trajectoryInfo;
    event::ConnectionPtr updateConnection;
    std::string trajectoryFile;
    std::vector<Waypoint> trajectory;
    common::Time startTime;
    size_t currentSegment = 0;
  };

  // Register the plugin with Gazebo
  GZ_REGISTER_MODEL_PLUGIN(TrajectoryFollowerPlugin)
}