# View_planning_simulator
This is the modified [view planning simulation system](https://github.com/psc0628/MA-SCVP).  
Follow the original one for installion.
## Prepare
Make sure "model_path" in DefaultConfiguration.yaml contains processed 3D models.  
You can find our processed data from [4D Plant Registration](https://www.ipb.uni-bonn.de/data/4d-plant-registration/index.html) in 4d_plant_registration_fixed_models.zip.  
The "pre_path" in DefaultConfiguration.yaml is the results saving path.  
The "pcnbv_path", "sc_net_path", "nricp_path" should be the python paths.  
## Usage
The mode of the system should be input in the Console.  
Then input 1 for combined pipeline and 0 for single method.  
Next input the method id you want to test. 
And if the method is search-based, you can input 1 for movement cost version and 0 otherwise.  
Finally give the object model names in the Console (-1 to break input).  
### Change in View Space
Set num_of_views to 32 and has_table to 1 for hemisphere.
Set num_of_views to 63 and has_table to 0 for sphere.
### Change in Inflation
Set add_current_inflation to 0 for the ablation.
### Mode 0
The system will genertate the ground truth point clouds of all visble voxels for all input objects. This will speed up evluation.  You should run this frist.
### Mode 2
The system will test all input objects by the input method with rotations and intital views.
#### Example Run Temporal-Prior-Guided View Planning
Input 2.
Input 1.
Input 9.
Input 0.
Input tomato_plant1_03-13.



