import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import calendar

Data = pd.read_csv("/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/carclaims.csv")

# Data[Data['MonthClaimed']=='0'].index
# processed_data = Data.drop([Data.index[1516]])

processed_data = Data

yLabel = processed_data['FraudFound']

# Change the attributes of make
processed_data = processed_data.replace({'Make':
    {
        'Accura': 'Make1',
        'BMW': 'Make1',
        'Ferrari': 'Make1',
        'Jaguar': 'Make1',
        'Lexus': 'Make1',
        'Mecedes': 'Make1',
        'Nisson': 'Make1',
        'Toyota': 'Make1',
        'Mazda': 'Make1',
        'Chevrolet': 'Make2',
        'Dodge': 'Make2',
        'VW':'Make2',
        'Honda':'Make2',
        'Ford':'Make2',
        'Pontiac': 'Make3',
        'Saturn': 'Make3',
        'Porche': 'Make4',
        'Mercury':'Make4',
        'Saab': 'Make4'
    }
})

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

# change Days:Policy Accident
processed_data = processed_data.replace({'Days:Policy-Accident':
    {
        '1 to 7': '1 to 15',
        '8 to 15': '1 to 15',
        'none': 'more than 30'
        # '15 to 30',
        # 'more than 30',
    }
})

# change AgeOfVehicle
processed_data = processed_data.replace({'AgeOfVehicle':
    {
        'new': 'less than 4 years',
        '2 years': 'less than 4 years',
        '3 years': 'less than 4 years',
        '4 years': 'less than 4 years',
        '5 years': '4 to 6 years',
        '6 years': '4 to 6 years',
        '7 years': '4 to 6 years',
        # 'more than 7'
    }
})

# change AddressChange-Claim
processed_data = processed_data.replace({'AddressChange-Claim':
    {
        'under 6 months': '0 to 3 years',
        '1 year': '0 to 3 years',
        '2 to 3 years': '0 to 3 years'
        # 'no change', '4 to 8 years',
    }
})

# change NumberOfCars
processed_data = processed_data.replace({'NumberOfCars':
    {
        # '1 vehicle',
        '2 vehicles': 'more than 1',
        '3 to 4': 'more than 1',
        '5 to 8': 'more than 1',
        'more than 8': 'more than 1'
    }
})

processed_data = processed_data.replace({'Age':{0:17}})

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
processed_data.drop(['PolicyType'], axis=1, inplace=True)
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

processed_data.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/Pre-Processed.csv", index=False)
processed_data.columns


###############################################
# Here onwards we will perform one-hot encoding
###############################################

processed_data_encoding = processed_data
processed_data_encoding
#select all the attributes of type object
carObject= processed_data_encoding.select_dtypes(include=['object']).copy()

#drop the attributes of type object
processed_data_encoding.drop(processed_data.select_dtypes(['object']),inplace=True,axis=1)
processed_data_encoding.head()

#normalization of feature to bring the value in the range [0,1]
minMaxScale= MinMaxScaler() #minMax scaler
processed_data_encoding = minMaxScale.fit_transform(processed_data_encoding)

#converting numpyarry to dataframe
processed_data_encoding = pd.DataFrame(processed_data_encoding)
processed_data_encoding.head()

onehot_encode_cols = ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault', 'VehicleCategory','VehiclePrice', 'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims','AgeOfVehicle', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments','AddressChange-Claim', 'NumberOfCars', 'BasePolicy']

processed_data_encoding_final = pd.get_dummies(carObject, prefix_sep="_",columns=onehot_encode_cols)
#Dummy encoding done?

processed_data_encoding_final

processed_data_encoding_final['Age']= processed_data_encoding[0]
processed_data_encoding_final['Deductible']= processed_data_encoding[1]
processed_data_encoding_final['DriverRating']= processed_data_encoding[2]
processed_data_encoding_final['DaysDiff']= processed_data_encoding[3]

processed_data_encoding_final['FraudFound'] = yLabel
processed_data_encoding_final.columns
processed_data_encoding_final.to_csv(r"/Users/abhiishekchugh/Documents/GitHub/CANN-for-Fraud-Detection/Automobile Insurance/data/Pre-Processed_OneHotEncoding.csv", index=False)





