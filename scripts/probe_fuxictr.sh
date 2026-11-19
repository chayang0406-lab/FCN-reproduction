P=/usr/local/lib/python3.11/site-packages/fuxictr
echo "=== pytorch dir ==="
ls $P/pytorch
echo "=== models.py BaseModel methods ==="
grep -nE "^class |^    def " $P/pytorch/models.py | head -60
echo "=== torch_utils.py ==="
grep -nE "^def |^class " $P/pytorch/torch_utils.py | head -30
echo "=== FeatureEmbedding ==="
grep -rn "class FeatureEmbedding" $P/pytorch/

