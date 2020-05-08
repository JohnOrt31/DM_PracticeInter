# -*- coding: utf-8 -*-
"""
@author: Jonathan Joel Corona Ortega
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



w_d = 'C:/Users/Jonathan/Desktop/PracticaMineria/'
i_f = w_d + 'survey_results_public.csv'

data = pd.read_csv(i_f, encoding = 'utf-8')

# Encuentra los valores unicos de Gender
# data['Gender'].unique()
# data.head()
#filtered = data['Gender'] == 'Woman'
#data[filtered][['Country', 'ConvertedComp']]


""" Funciones Generales"""
# Devuelve True si encuentra el género, o un false si no lo encuentra
def contain_option(value, option):
    return value.find(option) != -1

#Devuelve una lista con los datos de una columna filtrados, sin repetir
def unique_values(data, col):
    f_data = data[data[col].notnull()]
    uniques = list(f_data[col].unique())
    return list(set(';'.join(uniques).split(';')))

def notnull_filter(data, *cols):
    # filter_not_nulls(df, 'Age', 'Gender', 'Country')
    f = data[cols[0]].notnull()
    # df['Age'].notnull()
    for col in cols[1:]:
        f = f & data[col].notnull()
    return data[f]


def annual_salary(col, col_2, dato):
    #col = 'Gender'
    #col_2 = 'ConvertedComp'
    keys = unique_values(data, col)
    f_data = data[data[col].notnull() & data[col_2].notnull()]
    values = []
    
    for unique in keys:
        # (filtered_data[f]['Gender'] == 'Woman;Man') 
        # Detecta los valores existentes/no faltantes
        # filtered_data = data[data['Gender'].notnull()]
        
        #print(unique)
        f = f_data[col].apply(contain_option, args = (unique,))
        values.append(f_data[f])
        
    val_col = {k:v for k, v in zip(keys, values)}
    
    five_summary(val_col, dato, col_2)
    mean_and_std(val_col, dato, col_2)
    boxplot_d(val_col, dato, col_2)
    # boxplot_d2(data, col_2, col)
  
    
def med_mean_std(col, col_2, dato):
    #col = 'LanguageWorkedWith'
    #col_2 = 'Age'
    keys = unique_values(data, col)
    f_data = data[data[col].notnull() & data[col_2].notnull()]
    values = []
    
    for unique in keys:
          
        #print(unique)
        f = f_data[col].apply(contain_option, args = (unique,))
        values.append(f_data[f])
        
    val_col = {k:v for k, v in zip(keys, values)}
    
    print('Median: ', val_col[dato][col_2].quantile(.50))
    mean_and_std(val_col, dato, col_2)


def five_summary(val_col, dato, col_2):
    print('Min: ', val_col[dato][col_2].min())
    print('Max: ', val_col[dato][col_2].max())
    print('1st quartile: ', val_col[dato][col_2].quantile(.25))
    print('Median: ', val_col[dato][col_2].quantile(.50))
    print('3rd quartile: ', val_col[dato][col_2].quantile(.75))
 

def mean_and_std(val_col, dato, col_2):
    print('Mean: ',val_col[dato][col_2].mean())
    print('Standard Desviation:',val_col[dato][col_2].std())
 
    
def boxplot_d(val_col, dato, col_2):
    x = val_col[dato][col_2]
    plt.boxplot(x=x, notch=False, sym = '')

def boxplot_d2(data, column_1, column_2):

    new_data = notnull_filter(data, column_2, column_1)
    
    values = unique_values(new_data, column_2) 
    num = len(values)
    
    plt.figure(figsize = (6,6))
    for i, value in enumerate(values):
        filter_data = new_data[column_2].apply(contain_option, args = (value,))
        f_data = new_data[filter_data]
        #subplot(nrows, ncols, index, **kwargs)
        plt.subplot(num,3, i+1)
        plt.xlabel(column_1)
        plt.ylabel('Amount')
        plt.title(value[:10])
        plt.boxplot(f_data[column_1], notch=False, sym = '')
        plt.tight_layout()
    
def barplot_data(column):
    #column = 'DevType'
    new_data = notnull_filter(data, column)
    col_values = unique_values(data, column)

    frequencies= {} #keys = column values , values = Frequencies

    for col_value  in col_values:
        frequencie = sum(new_data[column].apply(contain_option, args=(col_value,)))
        frequencies[col_value] = frequencie
    
    height = np.arange(len(frequencies))
    plt.figure(figsize = (14,7))
    plt.bar(height = list(frequencies.values()), x = height, color = 'orange')
    plt.xticks(height, frequencies.keys(), rotation = 90)
    

def histograms_data(data, column_1, column_2, nrows=1, ncols=None, xlabel=None, ylabel=None, filename=None):
    #column_1 = 'YearsCode'
    #column_2 = 'Gender'
    new_data = notnull_filter(data, column_2, column_1)
    #new_data['YearsCode'].unique()
    
    values = unique_values(new_data, column_2) 
    if not ncols:
        ncols = len(values)
    if ylabel is None:
        ylabel = "Amount"
    if not xlabel:
        xlabel = column_1
    
    plt.figure(figsize = (14,8))
    for i, value in enumerate(values):
        filter_data = new_data[column_2].apply(contain_option, args = (value,))
        f_data = new_data[filter_data]
        #subplot(nrows, ncols, index, **kwargs)
        plt.subplot(nrows,ncols,i+1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(value[:10])
        plt.hist(f_data[column_1], bins = 10, color = 'purple')
        plt.tight_layout()
    plt.show()

def correlation_pearson(data, column1, column2):
    #column1 = 'YearsCode'
    #column2 = 'Convertedcomp'
    new_data = notnull_filter(data, column2, column1)
    
    x = new_data[column2].to_numpy()
    y = new_data[column1].to_numpy()
    #corr = np.corrcoef(x=x, y=y)
    #print(corr)
    plt.figure(figsize = (9,9), facecolor = 'w')
    plt.scatter(x=x, y=y)
    plt.xlabel(column2)
    plt.ylabel(column1)
    
    
    x_list = list(x)
    y_list = list(y)
    
    media1 = sum(x_list)/len(x_list)
    media2 = sum(y_list)/len(y_list)
    
    aux1 = aux2 = aux3 = 0
    
    for i in range(0, len(x_list)):
        aux1 +=  (x_list[i] - media1) * (y_list[i] - media2)   
        aux2 += (x_list[i] - media1) ** 2  
        aux3 += (y_list[i] - media2) ** 2
        
    res = aux1 / ( (aux2 ** (1/2)) * (aux3 ** (1/2)) )
    
    print('Correlation: ', res)
    





    
    
    
""" Ejercicios """
# 1. Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per gender.
def salary_gender(gender):
    print(gender)
    annual_salary('Gender', 'ConvertedComp', gender)
    print('\n')
    
# Metodo 1
#salary_gender('Man')
#salary_gender('Woman')
#salary_gender('Non-binary, genderqueer, or gender non-conforming')

# Método 2
keyGender = unique_values(data,'Gender')   
print('Exercise #1\n')
for i in keyGender:
    salary_gender(i)
    
    


# 2. Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per ethnicity.
def salary_ethnicity(ethnicity):
    print(ethnicity)
    annual_salary('Ethnicity', 'ConvertedComp', ethnicity)
    print('\n')
    
print('Exercise #2\n')
# Metodo 1
#salary_ethnicity('White or of European descent')
#salary_ethnicity('Middle Eastern')
#salary_ethnicity('Biracial')
#salary_ethnicity('Black or of African descent')
#salary_ethnicity('Multiracial')
#salary_ethnicity('White or of European descent')
#salary_ethnicity('Hispanic or Latino/Latina')
#salary_ethnicity('South Asian')
#salary_ethnicity('East Asian')

# Método 2
keyEthnicity = unique_values(data,'Ethnicity')   
print('Exercise #2\n')
for i in keyEthnicity:
    salary_ethnicity(i)


 
# 3. Compute the five-number summary, the boxplot, the mean, and the standard deviation for the annual salary per developer type.
def salary_devtype(devtype):
    print(devtype)
    annual_salary('DevType', 'ConvertedComp', devtype)
    print('\n')
    
# Método 1 (Pruebas)
#salary_devtype('Student')
#salary_devtype('Developer, front-end')
#salary_devtype('Developer, desktop or enterprise applications')
#salary_devtype('Developer, mobile')
#salary_devtype('Developer, back-end')
#salary_devtype('Database administrator')
#salary_devtype('Engineer, site reliability')
#salary_devtype('DevOps specialist')
#salary_devtype('Data scientist or machine learning specialist')
#salary_devtype('Product manager')
#salary_devtype('Senior executive/VP')
#salary_devtype('Scientist')
#salary_devtype('Developer, full-stack')
#salary_devtype('System administrator')
#salary_devtype('Developer, game or graphics')
#salary_devtype('Developer, embedded applications or devices')
#salary_devtype('Engineering manager')
#salary_devtype('Engineer, data')
#salary_devtype('Academic researcher')
#salary_devtype('Data or business analyst')
#salary_devtype('Marketing or sales professional')
#salary_devtype('Educator')
#salary_devtype('Designer')
#salary_devtype('Developer, QA or test')

         
# Método 2
keyDev = unique_values(data,'DevType')   
print('Exercise #3\n')
for i in keyDev:
    salary_devtype(i)
    
# 4. Compute the median, mean and standard deviation of the annual salary per country.
def salary_country(country):
    print(country)
    med_mean_std('Country', 'ConvertedComp', country)
    print('\n')
    
keyCountry = unique_values(data,'Country')   
print('Exercise #4\n')
for i in keyCountry:
    salary_country(i)
    

# 5. Obtain a bar plot with the frequencies of responses for each developer type.
def barplot_devtype():
    column = 'DevType'
    barplot_data(column)
barplot_devtype()

# 6. Plot histograms with 10 bins for the years of experience with coding per gender.    
def hist_exp_gender():
    data['YearsCode'].replace('Less than 1 year', '0.5', inplace=True)
    data['YearsCode'].replace('More than 50 years', '51', inplace=True)
    data['YearsCode'] =  data['YearsCode'].astype('float64')
    #data.dtypes
    histograms_data(data, 'YearsCode', 'Gender', xlabel='Experience', ylabel='', nrows=4, ncols=6)
hist_exp_gender()
    
        
        
# 7. Plot histograms with 10 bins for the average number of working hours per week, per developer type.  
def hist_hours_devtype():
    new_data = notnull_filter(data, 'WorkWeekHrs', 'DevType')
    f_data = (new_data['WorkWeekHrs'] > 30 ) & (new_data['WorkWeekHrs'] <80 )
    new_data = new_data[f_data]
    histograms_data(new_data, 'WorkWeekHrs', 'DevType', xlabel='Hours', ylabel='', nrows=4, ncols=6)
hist_hours_devtype()       
             
        
# 8. Plot histograms with 10 bins for the age per gender.
def hist_age_gender():
    new_data = notnull_filter(data, 'Age', 'Gender')
    f_data = (new_data['Age'] > 8 ) & (new_data['Age'] <80 )
    new_data = new_data[f_data]
    
    histograms_data(new_data, 'Age', 'Gender', xlabel='Edad', ylabel='', nrows=4, ncols=6)
hist_age_gender()       
    
    

# 9. Compute the median, mean and standard deviation of the age per programming language.
def age_proglenguage(progleng):
    print(progleng)
    med_mean_std('LanguageWorkedWith', 'Age', progleng)
    print('\n')
    
keyLenguage = unique_values(data,'LanguageWorkedWith')   
print('Exercise #9\n')
for i in keyLenguage:
    age_proglenguage(i)
    
    

# 10. Compute the correlation between years of experience and annual salary.
def corr_exp_salary():
    data['YearsCode'].replace('Less than 1 year', '0.5', inplace=True)
    data['YearsCode'].replace('More than 50 years', '51', inplace=True)
    data['YearsCode'] =  data['YearsCode'].astype('float64')
    correlation_pearson(data, 'YearsCode', 'ConvertedComp') 
corr_exp_salary()

    
    
# 11. Compute the correlation between the age and the annual salary.
def corr_age_salary():
    correlation_pearson(data, 'Age', 'ConvertedComp')
corr_age_salary()


# 12. Compute the correlation between educational level and annual salary. In this case, replace the string of the educational level by an ordinal index (e.g. Primary/elementary school = 1, Secondary school = 2, and so on).

edlevels = { 
     'I never completed any formal education' : 0,
     'Primary/elementary school' : 1,
     'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)' : 2,
     'Associate degree'  : 3,
     'Professional degree (JD, MD, etc.)' : 4,
     'Bachelor’s degree (BA, BS, B.Eng., etc.)' : 5,
     'Some college/university study without earning a degree' : 6,
     'Other doctoral degree (Ph.D, Ed.D., etc.)' : 7,
     'Master’s degree (MA, MS, M.Eng., MBA, etc.)': 8,
 }

def edLevel(item):
    return edlevels[item]


def correlation_edlevel_salary():
    new_data = notnull_filter(data, 'EdLevel', 'ConvertedComp')
    x = new_data['EdLevel'].apply(edLevel)
    y = new_data['ConvertedComp'].to_numpy()
    #corr = np.corrcoef(x=x, y=y)
    #print(corr)
    plt.figure(figsize = (9,9), facecolor = 'w')
    plt.scatter(x=x, y=y, color = 'g')
    plt.xlabel('EdLevel')
    plt.ylabel('ConvertedComp')
    
    
    x_list = list(x)
    y_list = list(y)
    
    media1 = sum(x_list)/len(x_list)
    media2 = sum(y_list)/len(y_list)
    
    aux1 = aux2 = aux3 = 0
    
    for i in range(0, len(x_list)):
        aux1 +=  (x_list[i] - media1) * (y_list[i] - media2)   
        aux2 += (x_list[i] - media1) ** 2  
        aux3 += (y_list[i] - media2) ** 2
        
    res = aux1 / ( (aux2 ** (1/2)) * (aux3 ** (1/2)) )
    
    print('Correlation: ', res)
correlation_edlevel_salary()

# 13. Obtain a bar plot with the frequencies of the different programming languages.
def barplot_proglenguage():
    column = 'LanguageWorkedWith'
    barplot_data(column)
barplot_proglenguage()













