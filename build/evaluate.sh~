#path="/home/jpl/TUM_Datasets/1Testing/f1_xyz/"
#para="/home/jpl/lines/TUM1.yaml"

#path="/home/jpl/TUM_Datasets/2Handheld/f1_desk/"
#para="/home/jpl/lines/TUM1.yaml"

#path="/home/jpl/TUM_Datasets/3Robot/f2_pioneer_slam/"
#para="/home/jpl/lines/TUM2.yaml"

#path="/home/jpl/TUM_Datasets/4Structure_vs_Texture/f3_structure_notexture_far/"
#para="/home/jpl/lines/TUM3.yaml"

path="/home/jpl/TUM_Datasets/6Reconstruction/f3_cabinet/"
para="/home/jpl/lines/TUM3.yaml"

#path="/home/jpl/TUM_Datasets/7Validation/f3_structure_notexture_near_validation/"
#para="/home/jpl/lines/TUM3.yaml"

./lineslam ${path} ${para}
#./naive_slam ${path} ${para}

python /home/jpl/TUM_Datasets/Tools/evaluate_ate.py --plot ate traject.txt ${path}groundtruth.txt

python /home/jpl/TUM_Datasets/Tools/evaluate_rpe.py --fixed_delta --plot rpe ${path}groundtruth.txt  traject.txt

#python ~/TUM_Datasets/Tools/associate.py rgb.txt depth.txt > associations.txt
