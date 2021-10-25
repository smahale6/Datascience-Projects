from pulp import *
import pandas as pd
import os


food_diet_raw = pd.read_excel('C://Users/D100793/Desktop/Hw 12/diet.xls')
food_diet = food_diet_raw[0:64].values.tolist()



foods = [i[0] for i in food_diet]
cost = dict([(i[0], float(i[1])) for i in food_diet])
calories = dict([(i[0], float(i[3])) for i in food_diet])
cholestrol = dict([(i[0], float(i[4])) for i in food_diet])
fat = dict([(i[0], float(i[5])) for i in food_diet])
sodium = dict([(i[0], float(i[6])) for i in food_diet])
carbs = dict([(i[0], float(i[7])) for i in food_diet])
fiber = dict([(i[0], float(i[8])) for i in food_diet])
protein = dict([(i[0], float(i[9])) for i in food_diet])
vitminA = dict([(i[0], float(i[10])) for i in food_diet])
vitminC = dict([(i[0], float(i[11])) for i in food_diet])
calcium = dict([(i[0], float(i[12])) for i in food_diet])
iron = dict([(i[0], float(i[13])) for i in food_diet])


minCalories = food_diet_raw['Calories'].iloc[65]
maxCalories = food_diet_raw['Calories'].iloc[66]

minCholesterol = food_diet_raw['Cholesterol mg'].iloc[65]     
maxCholesterol = 	food_diet_raw['Cholesterol mg'].iloc[66]

minTotal_Fat	= food_diet_raw['Total_Fat g'].iloc[65]
maxTotal_Fat	= food_diet_raw['Total_Fat g'].iloc[66]

minSodium = food_diet_raw['Sodium mg'].iloc[65]
maxSodium = food_diet_raw['Sodium mg'].iloc[66]

minCarbohydrates = food_diet_raw['Carbohydrates g'].iloc[65]
maxCarbohydrates = food_diet_raw['Carbohydrates g'].iloc[66]

minDietary_Fiber = food_diet_raw['Dietary_Fiber g'].iloc[65]
maxDietary_Fiber = food_diet_raw['Dietary_Fiber g'].iloc[66]

minProtein = food_diet_raw['Protein g'].iloc[65]
maxProtein	 = food_diet_raw['Protein g'].iloc[66]

minVit_A = food_diet_raw['Vit_A IU'].iloc[65]
maxVit_A = food_diet_raw['Vit_A IU'].iloc[66]

minVit_C = food_diet_raw['Vit_C IU'].iloc[65]
maxVit_C = food_diet_raw['Vit_C IU'].iloc[66]

minCalcium	 = food_diet_raw['Calcium mg'].iloc[65]
maxCalcium	 = food_diet_raw['Calcium mg'].iloc[66]

minIron = 	food_diet_raw['Iron mg'].iloc[65]
maxIron = 	food_diet_raw['Iron mg'].iloc[66]

food_diet_OF = LpProblem("Diet Optimization",LpMinimize)

#set the initial variables
foodVars = LpVariable.dicts("Foods", foods, lowBound = 0 )
chosenVars = LpVariable.dicts("Chosen", foods, lowBound = 0, upBound = 1, cat = "Binary")

#Add the objective function to mimimize the total cost
food_diet_OF += lpSum([cost[f]*foodVars[f] for f in foods]), "Total Cost"

#Add in the constraints
food_diet_OF += lpSum([calories[f]*foodVars[f] for f in foods]) >= minCalories, 'min Calories'
food_diet_OF += lpSum([calories[f]*foodVars[f] for f in foods]) <= maxCalories, 'max Calories'

food_diet_OF += lpSum([cholestrol[f]*foodVars[f] for f in foods]) >= minCholesterol, 'min Cholesterol'
food_diet_OF += lpSum([cholestrol[f]*foodVars[f] for f in foods]) <= maxCholesterol, 'max Cholesterol'

food_diet_OF += lpSum([fat[f]*foodVars[f] for f in foods]) >= minTotal_Fat, 'min fat'
food_diet_OF += lpSum([fat[f]*foodVars[f] for f in foods]) <= maxTotal_Fat, 'max fat'

food_diet_OF += lpSum([sodium[f]*foodVars[f] for f in foods]) >= minSodium, 'min sodium'
food_diet_OF += lpSum([sodium[f]*foodVars[f] for f in foods]) <= maxSodium, 'max sodium'

food_diet_OF += lpSum([carbs[f]*foodVars[f] for f in foods]) >= minCarbohydrates, 'min Carbs'
food_diet_OF += lpSum([carbs[f]*foodVars[f] for f in foods]) <= maxCarbohydrates, 'max Carbs'

food_diet_OF += lpSum([fiber[f]*foodVars[f] for f in foods]) >= minDietary_Fiber, 'min fiber'
food_diet_OF += lpSum([fiber[f]*foodVars[f] for f in foods]) <= maxDietary_Fiber, 'max fiber'

food_diet_OF += lpSum([protein[f]*foodVars[f] for f in foods]) >= minProtein, 'min protein'
food_diet_OF += lpSum([protein[f]*foodVars[f] for f in foods]) <= maxProtein, 'max protein'

food_diet_OF += lpSum([vitminA[f]*foodVars[f] for f in foods]) >= minVit_A, 'min vitA'
food_diet_OF += lpSum([vitminA[f]*foodVars[f] for f in foods]) <= maxVit_A, 'max vitA'

food_diet_OF += lpSum([vitminC[f]*foodVars[f] for f in foods]) >= minVit_C, 'min vitC'
food_diet_OF += lpSum([vitminC[f]*foodVars[f] for f in foods]) <= maxVit_C, 'max vitC'

food_diet_OF += lpSum([calcium[f]*foodVars[f] for f in foods]) >= minCalcium, 'min calcium'
food_diet_OF += lpSum([calcium[f]*foodVars[f] for f in foods]) <= maxCalcium, 'max calcium'

food_diet_OF += lpSum([iron[f]*foodVars[f] for f in foods]) >= minIron, 'min iron'
food_diet_OF += lpSum([iron[f]*foodVars[f] for f in foods]) <= maxIron, 'max iron'


        
for f in foods:
     food_diet_OF += foodVars[f] <= 10000000*chosenVars[f]
     food_diet_OF += foodVars[f] >= .1*chosenVars[f]

food_diet_OF += chosenVars['Frozen Broccoli'] + chosenVars['Celery, Raw'] <=1

food_diet_OF += chosenVars['Tofu'] + chosenVars['Roasted Chicken'] + \
chosenVars['Poached Eggs']+chosenVars['Scrambled Eggs']+chosenVars['Bologna,Turkey'] \
+chosenVars['Frankfurter, Beef']+chosenVars['Ham,Sliced,Extralean'] \
+chosenVars['Kielbasa,Prk']+chosenVars['Hamburger W/Toppings'] \
+chosenVars['Hotdog, Plain']+chosenVars['Pork'] +chosenVars['Sardines in Oil'] \
+chosenVars['White Tuna in Water'] >= 3



food_diet_OF.solve()
print("Status:", LpStatus[food_diet_OF.status])
for v in food_diet_OF.variables():
    if v.varValue != 0.0: #Only print items that are not zero
        print(v.name, "=", v.varValue)

print ("Total Cost of food with additional constraints is $%.2f" % value(food_diet_OF.objective))
