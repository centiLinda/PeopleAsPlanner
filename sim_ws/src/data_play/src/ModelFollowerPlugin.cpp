#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class ModelFollowerPlugin : public ModelPlugin
  {
  private:
    physics::ModelPtr model;
    physics::ModelPtr actor;
    event::ConnectionPtr updateConnection;
    std::string actorModelName;

  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      if (_sdf->HasElement("actor_name"))
        this->actorModelName = _sdf->Get<std::string>("actor_name");
      this->actor = _model->GetWorld()->ModelByName(this->actorModelName);  // Replace with your actor's name
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&ModelFollowerPlugin::OnUpdate, this));
    }

    void OnUpdate()
    {
      if (this->actor)
      {
        // Get actor's pose
        ignition::math::Pose3d actorPose = this->actor->WorldPose();

        // Keep Z height unchanged, only update X, Y, and Yaw
        ignition::math::Pose3d newPose(
          actorPose.Pos().X(),  // Keep X from the actor
          actorPose.Pos().Y(),  // Keep Y from the actor
          actorPose.Pos().Z()-0.5,        // Keep the original Z position
          actorPose.Rot().Roll(), 
          actorPose.Rot().Pitch(), 
          actorPose.Rot().Yaw()
        );

        // Set the model's pose with the fixed Z value
        this->model->SetWorldPose(newPose);
      }
    }
  };

  GZ_REGISTER_MODEL_PLUGIN(ModelFollowerPlugin)
}
