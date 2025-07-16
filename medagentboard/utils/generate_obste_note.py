from typing import Dict

import pandas as pd


def generate_obste_note(row: Dict):
    """
    根据单行患者数据（无论是单胎还是多胎）生成英文病历文本。
    """

    record = f"Patient ID: {row['PatientID']}\n"
    record += f"Maternal Age: {row['孕妇年龄']:.1f} years\n"
    record += f"Height: {row['孕妇身高']} cm\n"
    record += f"Pre-pregnancy Weight: {row['孕妇孕前体重']} kg\n"

    if pd.notna(row.get('孕前BMI')):
        record += f"Pre-pregnancy BMI: {row['孕前BMI']:.2f}\n"
    elif pd.notna(row.get('pre_bmi')):
        bmi_category = row['pre_bmi']
        if bmi_category == "体重过低": bmi_category = "Underweight"
        elif bmi_category == "体重正常": bmi_category = "Normal weight"
        elif bmi_category == "超重": bmi_category = "Overweight"
        elif bmi_category == "肥胖": bmi_category = "Obese"
        record += f"Pre-pregnancy BMI Category: {bmi_category}\n"

    record += "\nHistory:\n"
    history_items = []
    history_map = {
        '个人史_高血压史': "personal history of hypertension", '个人史_糖尿病史': "personal history of diabetes",
        '个人史_早产史': "personal history of preterm birth", '个人史_剖宫产史': "personal history of cesarean section",
        '个人史_巨大儿史': "personal history of macrosomia", '复发性流产史': "history of recurrent miscarriage",
        '家族史_高血压史': "family history of hypertension", '家族史_糖尿病史': "family history of diabetes"
    }
    for col, desc in history_map.items():
        if col in row and row[col] == 1:
            history_items.append(desc)

    if history_items:
        record += f"- {', '.join(history_items).capitalize()}.\n"
    else:
        record += "- No significant past medical or family history noted.\n"

    record += "\nCurrent Pregnancy:\n"
    record += f"- Last Menstrual Period: {row['末次月经日期']}\n"

    pregnancy_type_code = row.get('0单胎1双胎3三胎', 0) # 默认为单胎

    if pregnancy_type_code == 1:
        pregnancy_type = "Twin"
    elif pregnancy_type_code == 3:
        pregnancy_type = "Triplet"
    else:
        pregnancy_type = "Singleton"

    record += f"- This is a {pregnancy_type} pregnancy."
    if pregnancy_type == "Twin" and pd.notna(row.get('绒毛膜性')):
        chorionicity = "Dichorionic diamniotic" if row['绒毛膜性'] == '双绒毛膜双羊膜囊' else row['绒毛膜性']
        record += f" Chorionicity is {chorionicity}.\n"
    else:
        record += "\n"

    complications = []
    if row.get('妊娠期糖尿病') == 1: complications.append("gestational diabetes")
    if row.get('妊娠期高血压疾病') == 1: complications.append("hypertensive disorder of pregnancy")
    if row.get('选择性胎儿生长受限') == 1: complications.append("selective fetal growth restriction")
    if row.get('胎膜早破') == 1: complications.append("premature rupture of membranes (PROM)")
    if row.get('先兆早产') == 1: complications.append("threatened preterm labor")

    if complications:
        record += f"- Complicated by {', '.join(complications)}.\n"

    record += "\nDelivery Summary:\n"
    record += f"- Delivered at {row['分娩孕周_cal']:.1f} weeks of gestation on {row['分娩日期']}.\n"

    delivery_method = "Cesarean section" if row['分娩方式'] == '剖宫产' else "Vaginal delivery"
    rupture_type = "artificial" if row['破膜方式'] == '人工' else "spontaneous"

    record += f"- Mode of delivery: {delivery_method}.\n"
    record += f"- Rupture of membranes: {rupture_type}.\n"

    if pd.notna(row.get('单胎胎产式英文缩写')):
        record += f"- Fetal presentation: {row['单胎胎产式英文缩写']}.\n"
    if pd.notna(row.get('产后出血量（mL）')):
        record += f"- Estimated postpartum blood loss: {row['产后出血量（mL）']} mL.\n"

    record += "\nNeonate Information:\n"

    if pregnancy_type == 'Singleton':
        gender = "Female" if row.get('单胎性别') == '女' else "Male"
        outcome = "Live birth" if row.get('单胎胎儿结局') == '活产' else row.get('单胎胎儿结局')

        record += f"- Outcome: {outcome}, {gender}.\n"
        record += f"- Birth Weight: {row.get('单胎出生体重')}g, Birth Length: {row.get('单胎出生身长')}cm.\n"
        if pd.notna(row.get('单胎10分钟Apgar评分')):
            record += f"- 10-minute Apgar score: {int(row['单胎10分钟Apgar评分'])}.\n"

    elif pregnancy_type == 'Twin':
        record += "Twin 1:\n"
        record += f"- Outcome: {'Live birth' if row.get('双胎1胎儿结局') == '活产' else row.get('双胎1胎儿结局')}.\n"
        record += f"- Birth Weight: {row.get('双胎1出生体重')}g, Birth Length: {row.get('双胎1出生身长')}cm.\n"

        record += "\nTwin 2:\n"
        record += f"- Outcome: {'Live birth' if row.get('双胎2胎儿结局') == '活产' else row.get('双胎2胎儿结局')}.\n"
        record += f"- Birth Weight: {row.get('双胎2出生体重')}g, Birth Length: {row.get('双胎2出生身长')}cm.\n"
        if row.get('双胎2胎儿适于胎龄儿') == '二胎儿AGA':
            record += "- Assessed as Appropriate for Gestational Age (AGA).\n"

    return record


if __name__ == "__main__":
    df = pd.read_csv("my_datasets/ehr/obstetrics/raw/demo_solo.csv")
    for index, row in df.iterrows():
        print("--- Generating Record ---")
        record = generate_obste_note(row.to_dict())

    df = pd.read_csv("my_datasets/ehr/obstetrics/raw/demo_multi.csv")
    for index, row in df.iterrows():
        print("--- Generating Record ---")
        record = generate_obste_note(row.to_dict())