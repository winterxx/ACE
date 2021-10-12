import json, os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--input_path",
        help="input train/test splitting files",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--image_path",
        help="root path to save image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args

def convert(input_path, image_root):
    cls_idx = [ 86, 674, 566, 333, 417,  97, 755, 775, 984, 601, 402, 490, 870,
       976, 519, 318, 117, 304, 929,   2, 454, 548,  89, 219, 694, 444,
       951, 737, 173, 875, 882, 725, 329,  72, 545, 933, 613, 413, 924,
       112, 746,  18, 371, 554, 560, 603,  50, 398, 131, 182, 616,   3,
        68, 372,  46, 485, 830, 571, 460, 327, 532, 296, 805, 920, 594,
       763, 278, 559, 624, 309, 192, 597, 270, 806, 743, 608, 713,  67,
       665, 593, 765, 319, 480,  30, 834, 134, 922, 947, 340, 711, 257,
        85, 235,  28, 587, 408, 886, 172, 899, 787, 564, 687, 767, 968,
       404, 658, 591, 880, 578,   1,  29, 827, 126, 156, 191, 818, 280,
       119, 580, 876, 990, 449,  42, 987,  16, 487, 375, 518, 387, 828,
       783,  84, 464,  93, 788, 509, 589, 816, 978, 907, 913, 201, 803,
       654, 573, 302, 807, 652, 493, 565,  48, 176, 347, 132, 794, 352,
       752, 715, 862, 537,  52, 824, 148, 819, 476, 276, 284, 475,  90,
       468, 419, 739, 895, 709, 250,  21, 948, 595, 114, 496, 137,  22,
       844, 773, 950, 338, 762, 561,  57, 898, 923, 961, 244,  23, 213,
       388, 320, 896, 668, 600, 363, 549, 679, 253, 346, 995, 272, 820,
       396, 972, 442, 699, 557, 768, 247, 520, 785, 129, 414, 365, 858,
       435,  19,  55, 355, 133, 915, 473, 492, 540, 466, 441, 944,  61,
       904,   8, 174, 991, 423, 353, 348, 121, 555, 453, 832, 362, 315,
       985, 535,   4, 855, 361, 515, 934, 986, 717, 437, 890, 341, 628,
       229, 903, 672, 451, 279, 261,  74, 378, 757, 733, 177, 779,  83,
       477, 522,  26, 847, 879, 345, 802, 893, 107, 760, 336, 952, 609,
       269, 732, 586, 671, 799, 770, 367, 817, 325, 667, 108, 241, 357,
       486,  80, 786, 150, 240, 604, 569, 205, 472, 965, 415, 115, 374,
       281, 488, 394, 350, 508, 331,  14, 479, 617, 627, 311, 207, 245,
       161, 988, 412,  34, 224, 910, 528, 905, 106, 655, 469,  24, 212,
       856, 321, 447, 685, 958, 168, 443, 583, 892, 812, 103, 418, 567,
       206, 925, 906, 536, 426, 335, 465, 678, 969, 390, 675, 872, 635,
       194, 147, 503, 821, 662, 550, 891,  43, 686, 946, 724, 885, 262,
       501, 901, 810, 623, 392, 857, 585, 712, 165, 227, 531, 175, 158,
       637, 953, 710, 602, 979, 263,  39, 784, 289, 391, 368,  53, 831,
       973, 530, 945, 964, 811,  81, 101, 729, 562,   5, 546, 220, 445,
       246, 198, 638, 642, 188, 640, 534, 563, 366,  20, 781, 216, 314,
       145, 332, 218, 648, 769, 456, 619, 526, 146, 801, 914, 386, 644,
       500, 208, 793, 541, 837,  31, 751, 406, 373, 682, 551, 510, 645,
       471, 433, 516, 700, 795, 722, 753,  91, 324, 489, 271, 164, 251,
       956, 612, 942,  40, 884,  76, 754, 323, 128, 116, 719, 130,  63,
       683, 326, 195, 154, 796, 621, 897, 861, 239, 139, 677, 187, 513,
       629,  98, 143, 380, 977, 706, 184, 826, 622, 657, 705, 275, 118,
       970,  69, 618, 954, 258, 290, 288, 524, 574, 552, 859, 428, 611,
       160, 178,  54, 874,  36, 420, 721, 310, 943, 955, 735, 670, 266,
       728, 495, 919, 109, 383, 599,  32, 864, 369, 639, 838, 123,  44,
       647, 863, 723, 630, 676, 798, 401, 939, 989, 252, 226, 294, 704,
       521, 104, 680,  92, 703, 836, 846, 264, 197, 135, 688, 136, 242,
       474,  75, 308, 581, 439,  62, 960, 631, 833,  27, 926, 243,  33,
       186, 222, 744,  25, 543, 659,  11, 967, 527,  96, 937, 849, 533,
       777, 215, 505,  45, 636, 151, 928, 249, 292, 291, 809, 772, 974,
       221, 421, 570, 377, 313, 233, 889, 312,   0, 232, 620, 203, 758,
         9, 111, 360, 183, 370, 690, 462, 452, 450, 225, 494, 334, 539,
       553, 113,   6, 411, 163, 425, 577, 305, 664, 850, 822, 975, 696,
       980, 497,   7, 254, 416, 940, 504,  66, 841, 511, 458,  77, 502,
       202, 138, 209, 295, 300, 930, 835, 742, 774,  35, 512, 610, 446,
       506, 908, 684, 179, 330, 299, 287, 525, 228, 484, 382, 297, 982,
       422, 814, 734, 745, 481, 909,  37, 843, 256, 217, 592, 605, 983,
       698, 185, 430, 663,  88, 517, 931, 316, 159, 572, 149, 255, 238,
       936, 701, 596, 918, 791,  71, 643, 459, 169, 529, 190, 110,  70,
       167, 994, 171,  13, 829, 764, 761,  59, 399, 429,  82,  10, 981,
       403, 641,  60, 693, 457, 199, 692, 661, 971, 888,  47, 750, 740,
       957, 782, 598, 381, 483, 259, 393, 105, 427,  64, 634, 878, 400,
       749, 359, 780, 851, 379, 868, 871, 470, 432, 153,  78, 651, 558,
       157, 344, 902, 236, 695,  49, 649, 691, 389, 273, 607, 673, 790,
       894,  95, 839, 714, 342, 653, 282, 405,  51, 789, 144, 716, 932,
       200, 162, 666, 626, 122, 681, 395, 736, 615, 997, 881, 845, 166,
       547, 478, 707,  12, 440, 927, 697,  99, 189, 214, 656, 949, 354,
       407, 351, 966, 125, 193, 141, 120, 499, 230, 448, 921, 727, 410,
       301, 584, 582, 625, 590, 804, 808, 998, 999, 748, 277, 771, 364,
       124, 825, 507, 938, 181, 900,  87, 963, 461, 877, 996, 916, 992,
       322, 813, 614, 293, 283, 306, 204, 941, 842, 588, 303, 248, 339,
       840, 959, 852, 962, 455, 815, 467, 542, 317, 434, 689,  56, 349,
       911, 142, 669, 702, 993, 797,  41, 883, 869,  79,  94, 343, 274,
       463, 579, 741, 436, 792, 102,  38, 718, 576, 575, 267, 265, 409,
        17, 912, 237, 720, 286, 778, 260, 285, 376, 210, 358, 211, 223,
       800, 140, 873, 776, 127, 731, 298, 660, 431, 438, 100, 853, 180,
       556, 544, 152, 196,  73, 307, 424, 633, 397, 632, 759, 766, 867,
       738, 854, 514, 482,  65, 708, 523, 866, 568, 730, 606, 328, 848,
       170, 917, 756, 823, 231, 935, 337,  15, 650, 646, 234, 384, 385,
       538, 498, 726, 887, 268, 860, 491,  58, 747, 865, 155, 356]
    train = open(os.path.join(input_path, 'ImageNet_LT_train.txt')).readlines()
    valid = open(os.path.join(input_path, 'ImageNet_LT_test.txt')).readlines()
    train_annos = []
    valid_annos = []
    print("Converting file {} ...".format(os.path.join(input_path, 'ImageNet_LT_train.txt')))
    idx = 0
    for info in tqdm(train):
        image, category_id = info.strip().split(' ')
        category_id = cls_idx.index(int(category_id))
        train_annos.append({"image_id": idx,
                          "category_id": int(category_id),
                          "fpath": os.path.join(image_root, image)
                            })
        idx += 1
    print("Converting file {} ...".format(os.path.join(input_path, 'ImageNet_LT_test.txt')))
    idx = 0
    for info in tqdm(valid):
        image, category_id = info.strip().split(' ')
        category_id = cls_idx.index(int(category_id))
        valid_annos.append({"image_id": idx,
                            "category_id": int(category_id),
                            "fpath": os.path.join(image_root, image)
                            })
        idx += 1
    num_classes = 1000
    return {"annotations": train_annos, "num_classes": num_classes}, {'annotations': valid_annos, "num_classes": num_classes}

if __name__ == "__main__":
    args = parse_args()
    train_annos, valid_annos = convert(args.input_path, args.image_path)
    print("Converted, Saveing converted file to {}".format(args.output_path))
    with open(os.path.join(args.output_path, 'ImageNetLT_train_reorder.json'), "w") as f:
        json.dump(train_annos, f)
    with open(os.path.join(args.output_path, 'ImageNetLT_val_reorder.json'), "w") as f:
        json.dump(valid_annos, f)