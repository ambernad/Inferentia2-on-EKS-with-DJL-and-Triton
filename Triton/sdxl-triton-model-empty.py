Last login: Sat Oct 14 15:34:34 on ttys002
abernads@b0be8345944b ~ % cd Downloads 
abernads@b0be8345944b Downloads % ssh -i "BernKeyPair-ohio.pem" ubuntu@ec2-3-137-148-128.us-east-2.compute.amazonaws.com

=============================================================================
       __|  __|_  )
       _|  (     /   Deep Learning Base Neuron AMI (Ubuntu 20.04)
      ___|\___|___|
=============================================================================

Welcome to Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-1044-aws x86_64v)

* Supported EC2 instances: Inf1, Inf2, Trn1, Trn1n.
Neuron driver version: 2.12.18.0

AWS Deep Learning AMI Homepage: https://aws.amazon.com/machine-learning/amis/
Developer Guide and Release Notes: https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html
Support: https://forums.aws.amazon.com/forum.jspa?forumID=263
For a fully managed experience, check out Amazon SageMaker at https://aws.amazon.com/sagemaker
Security scan reports for python packages are located at: /opt/aws/dlami/info/
=============================================================================

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage
=============================================================================
       __|  __|_  )
       _|  (     /   Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04)
      ___|\___|___|
=============================================================================

Welcome to Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-1044-aws x86_64v)

* Supported EC2 instances: Inf1, Inf2, Trn1, Trn1n.
* To activate pre-built pytorch environment for Inf2, Trn*, run: 'source /opt/aws_neuron_venv_pytorch/bin/activate'
* To activate pre-built pytorch environment for Inf1, run: 'source /opt/aws_neuron_venv_pytorch_inf1/bin/activate'
* Neuron driver version:: 2.12.18.0

AWS Deep Learning AMI Homepage: https://aws.amazon.com/machine-learning/amis/
Release Notes: https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html
Support: https://forums.aws.amazon.com/forum.jspa?forumID=263
For a fully managed experience, check out Amazon SageMaker at https://aws.amazon.com/sagemaker
Security scan reports for python packages are located at: /opt/aws/dlami/info/
=============================================================================

  System information as of Sat Oct 14 13:54:47 UTC 2023


 * Ubuntu Pro delivers the most comprehensive open source security and
   compliance features.

   https://ubuntu.com/aws/pro

Expanded Security Maintenance for Applications is not enabled.

9 updates can be applied immediately.
To see these additional updates run: apt list --upgradable

31 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm

New release '22.04.3 LTS' available.
Run 'do-release-upgrade' to upgrade to it.


3 updates could not be installed automatically. For more details,
see /var/log/unattended-upgrades/unattended-upgrades.log

Last login: Sat Oct 14 13:07:00 2023 from 45.14.97.66
ubuntu@ip-172-31-47-241:~$ 
ubuntu@ip-172-31-47-241:~$ ps -a
    PID TTY          TIME CMD
  10120 pts/0    00:00:00 vi
  10233 pts/2    00:00:00 ps
ubuntu@ip-172-31-47-241:~$ kill 10120
ubuntu@ip-172-31-47-241:~$ ps
    PID TTY          TIME CMD
  10219 pts/2    00:00:00 bash
  10235 pts/2    00:00:00 ps
ubuntu@ip-172-31-47-241:~$ ps -a
    PID TTY          TIME CMD
  10236 pts/2    00:00:00 ps
ubuntu@ip-172-31-47-241:~$ cd triton-mode-repo/sdxl/1
ubuntu@ip-172-31-47-241:~/triton-mode-repo/sdxl/1$ ls
__pycache__  model-blank.py  model.bckp  model.py  sdxl_neuron
ubuntu@ip-172-31-47-241:~/triton-mode-repo/sdxl/1$ vi model
ubuntu@ip-172-31-47-241:~/triton-mode-repo/sdxl/1$ vi model.py 
ubuntu@ip-172-31-47-241:~/triton-mode-repo/sdxl/1$ vi model.py 
ubuntu@ip-172-31-47-241:~/triton-mode-repo/sdxl/1$ vi model.py 

        
        device_ids = [0,1,2,3]
        self.stable_diffusion = NeuronStableDiffusionXLPipeline.from_pretrained(model_id, export=False, **input_shapes, device_ids=device_ids)
        print("load complete")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
    
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt = inp.as_numpy()[0][0].decode()
        total_time = 0
        start_time = time.time()
        print("start inference")
        image = self.stable_diffusion(prompt).images[0]
        total_time = time.time() - start_time

        # Generate image key with current timestamp up to milliseconds
        current_time = time.strftime("%Y%m%d%H%M%S%f")[:-3]
        image_key = f"generated_image_{current_time}.png"
        bucket_name = "abernads-sd2-images"

        # Save image to local file
        image_path = "image.png"
        image.save(image_path)

        # Upload image to S3 bucket
        s3 = boto3.client('s3')
        s3.upload_file(image_path, bucket_name, image_key)

        result = "Inference time: " + str(np.round(total_time, 2)) + ". Image saved to S3 bucket " + bucket_name + " with key " + image_key
        response = pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("result",result,)])
        responses.append(response)
        return responses
