echo ==============================================
echo ================= JSC OpenML =================
echo ==============================================

alkaid report \
openml/rtl/epoch=27883-val_acc=0.771-ebops=13285-val_loss=0.635 \
openml/rtl/epoch=60648-val_acc=0.768-ebops=7229-val_loss=0.646 \
openml/rtl/epoch=116234-val_acc=0.758-ebops=2571-val_loss=0.683 \
openml/rtl/epoch=136813-val_acc=0.750-ebops=1572-val_loss=0.708 \
openml/rtl/epoch=199491-val_acc=0.733-ebops=530-val_loss=0.831 \
-c 'latency(ns)' LUT DSP FF 'Fmax(MHz)'

if [ -d "jedi/rtl" ]; then
    echo ==============================================
    echo ================== JEDI ======================
    echo ==============================================
    alkaid report jedi/rtl/8-3 jedi/rtl/16-3 jedi/rtl/32-3 -c 'latency(ns)' LUT DSP FF 'Fmax(MHz)'

    python synth_time.py jedi/rtl/8-3 jedi/rtl/16-3 jedi/rtl/32-3

fi

echo ==============================================
echo ================= Linformer ==================
echo ==============================================

alkaid report \
linear_ml4ps_models/rtl/lin8part \
linear_ml4ps_models/rtl/lin16part \
linear_ml4ps_models/rtl/lin32part \
-c 'latency(ns)' LUT DSP FF 'Fmax(MHz)'

python synth_time.py \
linear_ml4ps_models/rtl/lin8part \
linear_ml4ps_models/rtl/lin16part \
linear_ml4ps_models/rtl/lin32part \

echo ==============================================
echo ================ Transformer =================
echo ==============================================

alkaid report \
mha_ml4ps_models/rtl/mha8part \
mha_ml4ps_models/rtl/mha16part \
mha_ml4ps_models/rtl/mha32part \
-c 'latency(ns)' LUT DSP FF 'Fmax(MHz)'

python synth_time.py \
mha_ml4ps_models/rtl/mha8part \
mha_ml4ps_models/rtl/mha16part \
mha_ml4ps_models/rtl/mha32part \