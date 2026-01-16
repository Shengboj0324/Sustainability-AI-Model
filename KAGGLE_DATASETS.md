# Kaggle Datasets for Sustainability AI Model Training

## 8 Real Datasets Configured

### Dataset 1: Recyclable and Household Waste Classification (Master - 30 Classes)
- **Kaggle Link**: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification
- **Path**: `/kaggle/input/recyclable-and-household-waste-classification/images`
- **Classes**: 30 detailed waste categories
- **Size**: Large scale, high quality images
- **Type**: master
- **Status**: ✅ Already configured

### Dataset 2: Garbage Classification (12 Classes)
- **Kaggle Link**: https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- **Path**: `/kaggle/input/garbage-classification/garbage_classification`
- **Classes**: 12 categories including battery, biological, glass variants, metal, plastic, paper, cardboard, clothes, shoes, trash
- **Type**: mapped_12
- **Status**: ✅ Already configured

### Dataset 3: Waste Classification Data (2 Classes)
- **Kaggle Link**: https://www.kaggle.com/datasets/techsash/waste-classification-data
- **Path**: `/kaggle/input/waste-classification-data/DATASET`
- **Classes**: Organic (O) and Recyclable (R)
- **Size**: 22,000+ images
- **Type**: mapped_2
- **Status**: ✅ Already configured

### Dataset 4: Garbage Dataset (10 Classes)
- **Kaggle Link**: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
- **Path**: `/kaggle/input/garbage-classification-v2`
- **Classes**: Metal, Glass, Biological, Paper, Battery, Trash, Cardboard, Shoes, Clothes, Plastic
- **Size**: 19,762 images
- **Distribution**: Metal(1077), Glass(3199), Biological(997), Paper(1788), Battery(944), Trash(960), Cardboard(1853), Shoes(1977), Clothes(5327), Plastic(2096)
- **Type**: mapped_10
- **Status**: ✅ NEW - Added

### Dataset 5: Garbage Classification (6 Classes)
- **Kaggle Link**: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- **Path**: `/kaggle/input/garbage-classification`
- **Classes**: Cardboard, Glass, Metal, Paper, Plastic, Trash
- **Size**: 2,467 images
- **Distribution**: Cardboard(393), Glass(491), Metal(400), Paper(584), Plastic(472), Trash(127)
- **Type**: mapped_6
- **Status**: ✅ NEW - Added

### Dataset 6: Garbage Images Dataset (6 Classes - Balanced)
- **Kaggle Link**: https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification
- **Path**: `/kaggle/input/garbage-dataset-classification`
- **Classes**: Plastic, Metal, Glass, Cardboard, Paper, Trash
- **Size**: 14,000+ images (2,300-2,500 per class)
- **Quality**: All images standardized to 256x256 pixels, RGB format, cleaned of duplicates
- **Type**: mapped_6
- **Status**: ✅ NEW - Added

### Dataset 7: WaRP - Waste Recycling Plant Dataset (Industrial)
- **Kaggle Link**: https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset
- **Path**: `/kaggle/input/warp-waste-recycling-plant-dataset`
- **Classes**: Industrial waste sorting plant images with plastic types (PET, HDPE, PVC, LDPE, PP, PS), metal, glass, paper, cardboard
- **Size**: 845 MB
- **Type**: industrial
- **Unique**: Real industrial sorting plant data
- **Status**: ✅ NEW - Added

### Dataset 8: Multi Class Garbage Classification Dataset
- **Kaggle Link**: https://www.kaggle.com/datasets/vishallazrus/multi-class-garbage-classification-dataset
- **Path**: `/kaggle/input/multi-class-garbage-classification-dataset`
- **Classes**: Multiple waste categories for smart waste management
- **Size**: 31 MB, 2,752 files
- **Type**: multiclass
- **Status**: ✅ NEW - Added

## Total Dataset Statistics

- **Total Datasets**: 8
- **Total Images**: 60,000+ images
- **Total Classes Mapped**: 30 unified classes
- **Total Size**: ~2 GB
- **Coverage**: Household waste, industrial waste, recyclables, organics, hazardous materials

## Kaggle Notebook Setup Instructions

1. Create a new Kaggle notebook
2. Enable GPU (T4 or P100)
3. Add all 8 datasets as data sources:
   - Click "Add Data" → Search for each dataset by name
   - Add all 8 datasets to your notebook
4. Run the training notebook

## Dataset Paths in Kaggle

When you add datasets to your Kaggle notebook, they will be available at:
- `/kaggle/input/recyclable-and-household-waste-classification/`
- `/kaggle/input/garbage-classification/`
- `/kaggle/input/waste-classification-data/`
- `/kaggle/input/garbage-classification-v2/`
- `/kaggle/input/garbage-dataset-classification/`
- `/kaggle/input/warp-waste-recycling-plant-dataset/`
- `/kaggle/input/multi-class-garbage-classification-dataset/`

## Class Mapping Strategy

All datasets are intelligently mapped to the 30-class master schema:
- Industrial plastic types (PET, HDPE, etc.) → plastic_food_containers
- Generic categories → Specific master classes
- Organic/biological → food_waste
- Glass variants → Appropriate glass categories
- Metal types → Appropriate metal categories

## Data Quality

All datasets are:
- Real-world images
- Verified and published on Kaggle
- Actively maintained
- High usability scores
- Diverse in source and quality

