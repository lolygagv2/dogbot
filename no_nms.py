from hailo_sdk_client import ClientRunner

  def compile_model(runner):
      # Load the model
      runner.load_har('/app/bestdogyolo6_optimized.har')

      # Disable NMS postprocessing
      runner.model.set_nms_postprocess(False)
      # OR try:
      # runner.model.disable_postprocess()

      # Compile
      runner.compile()

      return runner

  runner = ClientRunner()
  compile_model(runner)
  runner.save_har('/app/bestdogyolo6_no_nms.hef')