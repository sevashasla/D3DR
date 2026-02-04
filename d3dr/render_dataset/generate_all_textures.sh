SCENENET_ROOT="/home/sevashasla/Documents/blender_data/SceneNetData"
DIRS=("1Bathroom" "1Bedroom" "1Kitchen" "1Living-room" "1Office")

for dir in "${DIRS[@]}"
do
    files=$(ls $SCENENET_ROOT/$dir)
    for file in $files
    do
        if [[ ${file: -4} == ".mtl" ]]; then
            python3 add_textures_scenergbd.py --mtl_path="$SCENENET_ROOT/$dir/$file"
        fi
    done
done
