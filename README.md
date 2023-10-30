<br />
<p align="center">
  <p align="center">
    <img src="assets/images/logo.png"" alt="Logo" width="40%">
  </p>

<h1 align="center">AffordPose: A Large-scale Dataset of Hand-Object Interactions with Affordance-driven Hand Pose  </h1>

  <p align="center">
    <strong>ICCV, 2023</strong>
    <br />
    <a href="#"><strong>Juntao Jian<sup>1</sup></strong></a>
    ·
    <a href="#"><strong>Xiuping Liu<sup>1</sup></strong></a>
    ·
    <a href="https://manyili12345.github.io/"><strong>Manyi Li<sup>2,<b>&#9742</b></sup></strong></a>
    ·
    <a href="https://csse.szu.edu.cn/staff/ruizhenhu/"><strong>Ruizhen Hu<sup>3</sup></strong></a>
    ·
    <a href="#"><strong>Jian Liu<sup>4,<b>&#9742</b></sup></strong></a>
    </br>
    </br>
      <sup>1 </sup>Dalian University of Technology  &nbsp;&nbsp; 
      <sup>2 </sup>Shandong University <br />
      <sup>3 </sup>Shenzhen University  &nbsp;&nbsp; <sup>4 </sup>Tsinghua University
    </br>
    <sup>&#9742</sup> Corresponding author
  </p>

  <p align="center">
    <a href='https://openaccess.thecvf.com/content/ICCV2023/html/Jian_AffordPose_A_Large-Scale_Dataset_of_Hand-Object_Interactions_with_Affordance-Driven_Hand_ICCV_2023_paper.html'>
      <img src='https://img.shields.io/badge/ICCV23-PDF-green?style=plastic&logo=readthedocs&logoColor=brightgreen' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2309.08942' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/ArXiv-PDF-darkgoldenrod?style=plastic&logo=arXiv&logoColor=gold' alt='ArXiv PDF'>
    </a>
    <a href='https://affordpose.github.io/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=skyblue' alt='Project Page'>
    </a>
    <a href="https://www.youtube.com/embed/s89tlzoM_M0?si=vPYkGPsXz583ndXW" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-chocolate?style=plastic&logo=youtube&logoColor=orange' alt='Youtube Video'>
    </a>
    <a href='https://affordpose.github.io/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Dataset-Page-red?style=plastic&logo=owncloud&logoColor=tomato' alt='Project Page'>
    </a>
  </p>
</p>
<br />

<div class="col-md-8 col-md-offset-2">
  <table>
    </tr>
    <tr>
      <td><img src="assets/images/bottle.gif" class="img-responsive" alt="overview" width="90%" style="max-height: 450px; margin: 30px auto" /></td>
      <td><img src="assets/images/handlebottle.gif" class="img-responsive" alt="overview" width="90%" style="max-height: 450px; margin: 30px auto" /></td>
      <td><img src="assets/images/knife.gif" class="img-responsive" alt="overview" width="90%" style="max-height: 450px; margin: 30px auto" /></td>
      <td><img src="assets/images/dispenser.gif" class="img-responsive" alt="overview" width="90%" style="max-height: 450px; margin: 30px auto" /></td>
    </tr>
  </table>

# Download Datasets

1. Download the AffordPose datasets from the [AffordPose Project Page](https://affordpose.github.io/). You can download specific categories or all the data according to your needs. The data are saved with the path: `AffordPose/Object_class/Object_id/affordance/xxx.json`, look like:
  
   ```
    .
    └── AffordPose
        ├──bottle
        │   ├──3415
        │   │   ├──3415_Twist
        │   │   │   ├── 1.json
        │   │   │   ├── ...
        │   │   │   └── 28.json
        │   │   │
        │   │   └──3415_Wrap-grasp
        │   │       ├── 1.json
        │   │       ├── ...
        │   │       └── 28.json
        |   |
        |   └── ...
        |
        └── ...
   ```

2. The structure in [xxx.json](samples/data.json) file as follows:
   ```
    .
    ├── xxx.json
        ├── rhand_mesh            # the hand mesh
        ├── dofs                  # the joint configurations of the hand
        ├── rhand_trans           # the translation of the paml
        ├── rhand_quat            # the rotation of the paml
        ├── object_mesh           # the object mesh, and the verts are annotated with affordance label
        ├── trans_obj             # with the default value: (0,0,0)
        ├── quat_obj              # with the default value: (1,0,0,0)
        ├── afford_name           # the object affordance corresponding to the interaction
        └── class_name            # the object class
   ```
#  Data visualization
- If you want to visualize the hand mesh, a feasible way is to save the value of "rhand_mesh" from the [xxx.json](samples/data.json) as [xxx.obj](samples/hand_mesh.obj) file and visualize it in [MeshLab](https://github.com/cnr-isti-vclab/meshlab), which is also applies to [object mesh](samples/bottle_mesh.obj). 

- The hand model we use following the [obman dataset](https://github.com/hassony2/obman), which ports the [MANO](https://mano.is.tue.mpg.de/) hand model to [GraspIt!](http://graspit-simulator.github.io/) simulator. 

- We used [GraspIt!](http://graspit-simulator.github.io/) to collect [xxx.xml](./samples/data.xml) data and ran `ManoHand_xml2mesh.py` to obtain the [hand mesh](./samples/hand_mesh_mm.obj) in 'mm'. 
Please note that you cannot obtain the correct hand mesh in 'm' by simply changing the 'scale' parameter in this python file.
  ```Shell
  $ python ./ManoHand_xml2mesh.py --xml_path PATH_TO_DATA.xml --mesh_path PATH_TO_SAVE_DATA.obj --part_path DIRPATH_TO_SAVE_HAND_PARTS
  ```

# Citation

If you find AffordPose dataset is useful for your research, please considering cite us:

    @InProceedings{Jian_2023_ICCV,
      author    = {Jian, Juntao and Liu, Xiuping and Li, Manyi and Hu, Ruizhen and Liu, Jian},
      title     = {AffordPose: A Large-Scale Dataset of Hand-Object Interactions with Affordance-Driven Hand Pose},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      month     = {October},
      year      = {2023},
      pages     = {14713-14724}
    }
