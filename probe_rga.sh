H=/home/user/luckfox-pico/media/out/include/rga
echo "=== imcopy ==="
grep -rn "imcopy" $H | head -20
echo "=== imcvtcolor ==="
grep -rn "imcvtcolor" $H | head -10
echo "=== imsync/imcheck/imblit ==="
grep -rn "imsync\|IM_STATUS imcheck\|imblit\|improcess" $H | head -20
echo "=== im2d_version ==="
cat $H/im2d_version.h | head -30