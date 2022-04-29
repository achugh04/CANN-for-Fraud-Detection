import pandas as pd
import numpy as np
import calendar
import category_encoders as ce

Data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/carclaims.csv")


processed_data = Data

yLabel = processed_data['FraudFound']

processed_data.VehiclePrice.unique()
processed_data = processed_data.replace({'Age':{0:17}})

# change the vehicle price
processed_data = processed_data.replace({'VehiclePrice':
    {
        # 'less than 20,000',
        '20,000 to 29,000': '20,000 to 39,000',
        '30,000 to 39,000': '20,000 to 39,000',
        '40,000 to 59,000': '40,000 to 69,000',
        '60,000 to 69,000': '40,000 to 69,000',
        # 'more than 69,000'
    }
})
# function to calculate the no of days passed between the accident and the claims.
# Reporting Gap:

def get_date(year, month, weekOfMonth, dayOfWeek):
    count = 0
    c = calendar.TextCalendar(firstweekday=0)
    l = []
    for i in c.itermonthdates(year, month):
        l.append(i)
    for j in range(len(l)):
        day = calendar.day_name[l[j].weekday()]
        # print(j,l[j-2])
        if dayOfWeek ==0:
            return None
        if day == dayOfWeek:
            count += 1
            if count == weekOfMonth:
                # print('here',l[j])
                return l[j]


def differ_days(date1, date2):
    a = date1
    b = date2
    c = get_date(1994,1,1,"Saturday")
    if a!=None and b!=None:
        return (a - b).days
    elif(a==None):
        return (b-c).days
    else:
        return (a-c).days


# replace map
replace_Month = {'Month': {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                           'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}}

replace_MonthClaimed = {'MonthClaimed': {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}}

processed_data.replace(replace_Month, inplace=True)
processed_data.replace(replace_MonthClaimed, inplace=True)

processed_data["Month"] = pd.to_numeric(processed_data["Month"])
processed_data["MonthClaimed"] = pd.to_numeric(processed_data["MonthClaimed"])

day_diff = np.zeros((processed_data.shape[0], 1))
#processed_data[['Month','MonthClaimed']].fillna(6)
for i in range(processed_data.shape[0]):
    if (processed_data['MonthClaimed'][i] - processed_data['Month'][i]) < 0:
        year2 = processed_data['Year'][i] + 1
        month2 = processed_data['MonthClaimed'][i]
        week2 = processed_data['WeekOfMonthClaimed'][i]
        day2 = processed_data['DayOfWeekClaimed'][i]
        year1 = processed_data['Year'][i]
        month1 = processed_data['Month'][i]
        week1 = processed_data['WeekOfMonth'][i]
        day1 = processed_data['DayOfWeek'][i]
        if (month1==0):
            month1=6
        elif(month2==0):
            month2=6
        day_diff[i] = differ_days(get_date(year2, month2, week2, day2), get_date(year1, month1, week1, day1))
    else:
        year2 = processed_data['Year'][i]
        month2 = processed_data['MonthClaimed'][i]
        week2 = processed_data['WeekOfMonthClaimed'][i]
        day2 = processed_data['DayOfWeekClaimed'][i]
        year1 = processed_data['Year'][i]
        month1 = processed_data['Month'][i]
        week1 = processed_data['WeekOfMonth'][i]
        day1 = processed_data['DayOfWeek'][i]
        if (month1==0):
            month1=6
        elif(month2==0):
            month2=6
        day_diff[i] = differ_days(get_date(year2, month2, week2, day2), get_date(year1, month1, week1, day1))
    # print(i, day_diff[i])

# adding column to the existing dataframe
processed_data['daysDiff'] = day_diff
processed_data['daysDiff'][processed_data['daysDiff']<1] = 0

# now drop the original attibutes, like 'Month' column(we don't need anymore)
processed_data.drop(['Year'], inplace=True, axis=1)
processed_data.drop(['Month'], axis=1, inplace=True)
processed_data.drop(['MonthClaimed'], axis=1, inplace=True)
processed_data.drop(['WeekOfMonth'], inplace=True, axis=1)
processed_data.drop(['WeekOfMonthClaimed'], inplace=True, axis=1)
processed_data.drop(['DayOfWeek'], axis=1, inplace=True)
processed_data.drop(['DayOfWeekClaimed'], axis=1, inplace=True)

processed_data.drop(['PolicyNumber'], inplace=True, axis=1)
# processed_data.drop(['PolicyType'], axis=1, inplace=True)
processed_data.drop(['RepNumber'], axis=1, inplace=True)
processed_data.drop(['AgeOfPolicyHolder'], inplace=True, axis=1)

# Change the class attribute
processed_data = processed_data.replace({'FraudFound':
    {
        'Yes': 1,
        'No': 0
    }
})

# drop the label from the dataset
yLabel = processed_data['FraudFound']
processed_data.drop(['FraudFound'], inplace=True, axis=1)

processed_data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed_Chugh.csv", index=False)

yLabel.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed_Chugh_Target.csv", index=False)

processed_data.columns
encoding_cols = ['Make','AccidentArea','Sex','MaritalStatus','Fault','PolicyType','VehicleCategory','VehiclePrice','Days:Policy-Accident','PastNumberOfClaims','Days:Policy-Claim','AgeOfVehicle','PoliceReportFiled','WitnessPresent','AgentType','NumberOfSuppliments','AddressChange-Claim','NumberOfCars','BasePolicy']

###############################################
# Here onwards we will perform only Label Encoding on Baseline data
###############################################
data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/carclaims.csv")
data.drop(['FraudFound'], inplace=True, axis=1)
# for i in data.columns:
#     print(data[i].unique(),i)
from sklearn.preprocessing import LabelEncoder
for i in data.columns:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])
data=pd.concat([data,yLabel],axis=1)
data=data.rename({'daysDiff':'DaysDiff'},axis=1)
print(data.columns.size)
data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv",index=False)
data.columns



##############################################
# Here onwards we will perform only Binary Encoding
###############################################
data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed_Chugh.csv")
data.head().T
for i in encoding_cols:
    be = ce.BinaryEncoder(cols=[i])
    be_df = be.fit_transform(data[i])
    data.drop([i],inplace=True, axis=1)
    data = pd.concat([data,be_df],axis=1)

data.head().T
data=pd.concat([data,yLabel],axis=1)
data=data.rename({'daysDiff':'DaysDiff'},axis=1)
print(data.columns.size)
# Data['AddressChange-Claim'].unique()
data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Binary_Encoding.csv",index=False)
data.columns
# ###############################################
# # Here onwards we will perform only Ordinal Encoding
# ###############################################


data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed_Chugh.csv")
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
for i in encoding_cols:
    oe_val = oe.fit_transform(data[i].values.reshape(-1, 1))
    data.drop([i],inplace=True, axis=1)
    data = pd.concat([data,pd.DataFrame(oe_val,columns = [i])],axis=1)
    

data.head().T
data=pd.concat([data,yLabel],axis=1)
data=data.rename({'daysDiff':'DaysDiff'},axis=1)
print(data.columns.size)
# Data['AddressChange-Claim'].unique()
data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Ordinal_Encoding.csv",index=False)
data.columns
# ###############################################
# # Here onwards we will perform only Hermet Encoding
# ###############################################
data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed_Chugh.csv")


for i in encoding_cols:    
    he = ce.HelmertEncoder(cols=[i])
    he_df = he.fit_transform(data[i])
    he_df.drop(['intercept'],inplace=True, axis=1)
    data.drop([i], inplace=True, axis=1)
    data = pd.concat([data,he_df],axis=1)
    


data.head().T
data=pd.concat([data,yLabel],axis=1)
data=data.rename({'daysDiff':'DaysDiff'},axis=1)
print(data.columns.size)
# Data['AddressChange-Claim'].unique()
data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Hermet_Encoding.csv",index=False)
data.columns

# ###############################################
# # Here onwards we will perform only LOOE
# ###############################################



# ###############################################
# # Here onwards we will perform only LOOE
# ###############################################
# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv")

# for i in ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault', 'VehicleCategory','VehiclePrice', 'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims','AgeOfVehicle', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments','AddressChange-Claim', 'NumberOfCars', 'BasePolicy','PolicyType']:
#     loue = ce.LeaveOneOutEncoder(cols=[i])
#     data[i] = loue.fit_transform(data[i],yLabel)

# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_LOOE_FOR_ALL.csv",index=False)
# data.columns

# ###############################################
# # Here onwards we will perform only CatBooster Encoding
# ###############################################
# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv")
# for i in ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault', 'VehicleCategory','VehiclePrice', 'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims','AgeOfVehicle', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments','AddressChange-Claim', 'NumberOfCars', 'BasePolicy','PolicyType']:
#     cbe = ce.CatBoostEncoder(cols=[i])
#     data[i] = cbe.fit_transform(data[i],yLabel)

# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_CBE_FOR_ALL.csv",index=False)


###############################################
# Here onwards we will perform only WeightOfEvidence Encoding
# This encoding is used in the finance industry to detect fraud
###############################################
data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed_Chugh.csv")
for i in encoding_cols:
    woe = ce.WOEEncoder(cols=[i])
    data[i] = woe.fit_transform(data[i],yLabel)

data.head().T
data=pd.concat([data,yLabel],axis=1)
data=data.rename({'daysDiff':'DaysDiff'},axis=1)
print(data.columns.size)
# Data['AddressChange-Claim'].unique()
data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_WOE_FOR_ALL.csv",index=False)
data.columns

# ###############################################
# # Here onwards we will perform only LOOE
# ###############################################



# ###############################################
# # Here onwards we will perform only LOOE
# ###############################################
# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv")

# for i in ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault', 'VehicleCategory','VehiclePrice', 'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims','AgeOfVehicle', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments','AddressChange-Claim', 'NumberOfCars', 'BasePolicy','PolicyType']:
#     loue = ce.LeaveOneOutEncoder(cols=[i])
#     data[i] = loue.fit_transform(data[i],yLabel)

# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_LOOE_FOR_ALL.csv",index=False)
# data.columns

# ###############################################
# # Here onwards we will perform only CatBooster Encoding
# ###############################################
# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv")
# for i in ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault', 'VehicleCategory','VehiclePrice', 'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims','AgeOfVehicle', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments','AddressChange-Claim', 'NumberOfCars', 'BasePolicy','PolicyType']:
#     cbe = ce.CatBoostEncoder(cols=[i])
#     data[i] = cbe.fit_transform(data[i],yLabel)

# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_CBE_FOR_ALL.csv",index=False)


# ###############################################
# # Here onwards we will perform only WeightOfEvidence Encoding
# # This encoding is used in the finance industry to detect fraud
# ###############################################
# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv")
# for i in ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault', 'VehicleCategory','VehiclePrice', 'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims','AgeOfVehicle', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments','AddressChange-Claim', 'NumberOfCars', 'BasePolicy','PolicyType']:
#     woe = ce.WOEEncoder(cols=[i])
#     data[i] = woe.fit_transform(data[i],yLabel)

# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_WOE_FOR_ALL.csv",index=False)


# ###############################################
# # Here onwards we will perform >2 types of encoding
# ###############################################

# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv")
# # For the VehiclePrice and Days:Policy-Accident, AddressChange-Claim column we use Backward Difference Encoding
# for i in ['VehiclePrice', 'Days:Policy-Accident','AddressChange-Claim','BasePolicy']:
#     bde = ce.BackwardDifferenceEncoder(cols=[i])
#     bde_df = bde.fit_transform(data[i],yLabel)
#     bde_df.drop(['intercept'],inplace=True, axis=1)
#     data=pd.concat([data,bde_df],axis=1)
#     data.drop([i], inplace=True, axis=1)

# data.columns

# #Performing CatBoost encoding for Make, AgeOVehicle columns
# for i in ['Make','AgeOfVehicle']:
#     cbe = ce.CatBoostEncoder(cols=[i])
#     data[i] = cbe.fit_transform(data[i],yLabel)
    
# # Performing LOOE encoding for Month. DayOfWeek, DayOfWeekClaimed, MonthClaimed columns   
# for i in ['Month','DayOfWeek','DayOfWeekClaimed','MonthClaimed']:
#     loue = ce.LeaveOneOutEncoder(cols=[i])
#     data[i] = loue.fit_transform(data[i],yLabel)
    
# #Target encoding for PoliceReportFiled, WitnessPresentcolumns
# for i in ['PoliceReportFiled','WitnessPresent','AgeOfPolicyHolder']:
#     te = ce.TargetEncoder(cols=[i])
#     data[i] = te.fit_transform(data[i],yLabel)

# # for i in ['AgeOfPolicyHolder']:    
# #     hash_df = ce.HashingEncoder(cols=[i])
# #     hash_df = hash_df.fit_transform(data[i],yLabel)
# #     data=pd.concat([data,hash_df],axis=1)
# #     data.drop([i], inplace=True, axis=1)
    

# data.head().T
# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# # Data['AddressChange-Claim'].unique()
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh.csv",index=False)
# data.columns


# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_WOE_FOR_ALL.csv",index=False)


# ###############################################
# # Here onwards we will perform >2 types of encoding
# ###############################################

# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Label_Encoding.csv")
# # For the VehiclePrice and Days:Policy-Accident, AddressChange-Claim column we use Backward Difference Encoding
# for i in ['VehiclePrice', 'Days:Policy-Accident','AddressChange-Claim','BasePolicy']:
#     bde = ce.BackwardDifferenceEncoder(cols=[i])
#     bde_df = bde.fit_transform(data[i],yLabel)
#     bde_df.drop(['intercept'],inplace=True, axis=1)
#     data=pd.concat([data,bde_df],axis=1)
#     data.drop([i], inplace=True, axis=1)

# data.columns

# #Performing CatBoost encoding for Make, AgeOVehicle columns
# for i in ['Make','AgeOfVehicle']:
#     cbe = ce.CatBoostEncoder(cols=[i])
#     data[i] = cbe.fit_transform(data[i],yLabel)
    
# # Performing LOOE encoding for Month. DayOfWeek, DayOfWeekClaimed, MonthClaimed columns   
# for i in ['Month','DayOfWeek','DayOfWeekClaimed','MonthClaimed']:
#     loue = ce.LeaveOneOutEncoder(cols=[i])
#     data[i] = loue.fit_transform(data[i],yLabel)
    
# #Target encoding for PoliceReportFiled, WitnessPresentcolumns
# for i in ['PoliceReportFiled','WitnessPresent','AgeOfPolicyHolder']:
#     te = ce.TargetEncoder(cols=[i])
#     data[i] = te.fit_transform(data[i],yLabel)

# # for i in ['AgeOfPolicyHolder']:    
# #     hash_df = ce.HashingEncoder(cols=[i])
# #     hash_df = hash_df.fit_transform(data[i],yLabel)
# #     data=pd.concat([data,hash_df],axis=1)
# #     data.drop([i], inplace=True, axis=1)
    

# data.head().T
# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# # Data['AddressChange-Claim'].unique()
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh.csv",index=False)
# data.columns

# data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/carclaims.csv")
# data.drop(['FraudFound'], inplace=True, axis=1)
# # for i in data.columns:
# #     print(data[i].unique(),i)
# for i in data.columns:
#     woe = ce.WOEEncoder(cols=[i])
#     data[i] = woe.fit_transform(data[i],yLabel)
 

# data=pd.concat([data,yLabel],axis=1)
# data=data.rename({'daysDiff':'DaysDiff'},axis=1)
# print(data.columns.size)
# data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/pre-processing done/Pre-Processed-Encoded_Chugh_Baseline_Ordinal_Encoding.csv",index=False)
# data.columns

