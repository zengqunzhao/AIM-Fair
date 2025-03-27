from Text_Smiling_Male import *
from Text_Smiling_Young import *
from Text_BlondHair_Male import *
from Text_BlackHair_Young import *
from Text_Male_Young import *
from Text_Young_Male import *
from Text_Female_White import *
from Text_Smiling_MaleYoung import *
from Text_Cat_Dark import *

####################################################
smiling_male_1 = {
    "1_1": list_single_Smiling_Male,
    "1_0": list_single_Smiling_Female,
    "0_1": list_single_NoSmiling_Male,
    "0_0": list_single_NoSmiling_Female,
}
smiling_male_50 = {
    "1_1": list_Smiling_Male,
    "1_0": list_Smiling_Female,
    "0_1": list_NoSmiling_Male,
    "0_0": list_NoSmiling_Female,
}

smiling_young_1= {
    "1_1": list_single_Smiling_Young,
    "1_0": list_single_Smiling_Aged,
    "0_1": list_single_NotSmiling_Young,
    "0_0": list_single_NotSmiling_Aged,
}
smiling_young_50 = {
    "1_1": list_Smiling_Young,
    "1_0": list_Smiling_Aged,
    "0_1": list_NotSmiling_Young,
    "0_0": list_NotSmiling_Aged,
}

blond_hair_male_1 = {
    "1_1": list_single_BlondHair_Male,
    "1_0": list_single_BlondHair_Female,
    "0_1": list_single_NotBlondHair_Male,
    "0_0": list_single_NotBlondHair_Female,
}
blond_hair_male_50 = {
    "1_1": list_BlondHair_Male,
    "1_0": list_BlondHair_Female,
    "0_1": list_NotBlondHair_Male,
    "0_0": list_NotBlondHair_Female,
}

black_hair_young_1 = {
    "1_1": list_single_BlackHair_Young,
    "1_0": list_single_BlackHair_Aged,
    "0_1": list_single_NotBlackHair_Young,
    "0_0": list_single_NotBlackHair_Aged,
}
black_hair_young_50 = {
    "1_1": list_BlackHair_Young,
    "1_0": list_BlackHair_Aged,
    "0_1": list_NotBlackHair_Young,
    "0_0": list_NotBlackHair_Aged,
}

male_young_1 = {
    "1_1": list_single_Male_Young,
    "1_0": list_single_Male_Aged,
    "0_1": list_single_Female_Young,
    "0_0": list_single_Female_Aged,
}
male_young_50 = {
    "1_1": list_Male_Young,
    "1_0": list_Male_Aged,
    "0_1": list_Female_Young,
    "0_0": list_Female_Aged,
}

young_male_1 = {
    "1_1": list_single_Young_Male,
    "1_0": list_single_Young_Female,
    "0_1": list_single_Aged_Male,
    "0_0": list_single_Aged_Female,
}
young_male_50 = {
    "1_1": list_Young_Male,
    "1_0": list_Young_Female,
    "0_1": list_Aged_Male,
    "0_0": list_Aged_Female,
}

female_white_50 = {
    "1_1": list_Female_White,
    "1_0": list_Female_NonWhite,
    "0_1": list_Male_White,
    "0_0": list_Male_NonWhite,
}

smiling_male_young_50 = {
    "1_1_1": list_Smiling_Male_Young,
    "1_1_0": list_Smiling_Male_Aged,
    "1_0_1": list_Smiling_Female_Young,
    "1_0_0": list_Smiling_Female_Aged,
    "0_1_1": list_NoSmiling_Male_Young,
    "0_1_0": list_NoSmiling_Male_Aged,
    "0_0_1": list_NoSmiling_Female_Young,
    "0_0_0": list_NoSmiling_Female_Aged,
}

cat_bright_50 = {
    "1_1": list_Cat_Bright,
    "1_0": list_Cat_Dark,
    "0_1": list_Dog_Bright,
    "0_0": list_Dog_Dark,
}

####################################################
####################################################
all_prompt_list = {
    #### Smiling_Male
    "Smiling_Male_Single_Prompt": smiling_male_1,
    "Smiling_Male_Multiply_Prompt": smiling_male_50,
    #### Smiling_Young
    "Smiling_Young_Single_Prompt": smiling_young_1,
    "Smiling_Young_Multiply_Prompt": smiling_young_50,
    #### BlondHair_Male
    "BlondHair_Male_Single_Prompt": blond_hair_male_1,
    "BlondHair_Male_Multiply_Prompt": blond_hair_male_50,
    #### BlackHair_Young
    "BlackHair_Young_Single_Prompt": black_hair_young_1,
    "BlackHair_Young_Multiply_Prompt": black_hair_young_50,
    #### Male_Young
    "Male_Young_Single_Prompt": male_young_1,
    "Male_Young_Multiply_Prompt": male_young_50,
    #### Young_Male
    "Young_Male_Single_Prompt": young_male_1,
    "Young_Male_Multiply_Prompt": young_male_50,

    #### Female_White
    "Female_White_Multiply_Prompt": female_white_50,

    #### Smiling_Male_Young
    "Smiling_MaleYoung_Multiply_Prompt": smiling_male_young_50,

    #### Cat_Bright
    "Cat_Bright_Multiply_Prompt": cat_bright_50

}
####################################################