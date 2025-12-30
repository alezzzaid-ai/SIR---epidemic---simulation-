import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# STEP 1 — SIR EPIDEMIC MODEL
# -------------------------------

N = 1500
I0 = 20
BETA = 0.35
GAMMA = 0.15
DAYS = 65

S = N - I0
I = I0
R = 0

S_data = [S]
I_data = [I]
R_data = [R]

for day in range(DAYS):
    new_infected = BETA * S * I / N
    new_recovered = GAMMA * I

    S -= new_infected
    I += new_infected - new_recovered
    R += new_recovered

    S_data.append(S)
    I_data.append(I)
    R_data.append(R)

# -------------------------------
# STEP 2 — VISUALIZATION
# -------------------------------

plt.figure(figsize=(10, 6))
plt.plot(S_data, label="Susceptible")
plt.plot(I_data, label="Infected")
plt.plot(R_data, label="Recovered")
plt.xlabel("Days")
plt.ylabel("Population")
plt.title("SIR Epidemic Simulation")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# STEP 3 — ML PREDICTION
# -------------------------------

days = np.array(range(len(I_data))).reshape(-1, 1)
infected = np.array(I_data)

model = LinearRegression()
model.fit(days, infected)

future_days = np.array(range(len(I_data) + 15)).reshape(-1, 1)
predicted = model.predict(future_days)

plt.figure(figsize=(10, 6))
plt.plot(days, infected, label="Actual Data")
plt.plot(future_days, predicted, "--", label="Predicted Trend")
plt.xlabel("Days")
plt.ylabel("Infected")
plt.title("ML Prediction of Infection Trend")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# STEP 4 — MODEL EVALUATION
# -------------------------------

split_index = int(0.8 * len(days))

X_train = days[:split_index]
X_test = days[split_index:]
y_train = infected[:split_index]
y_test = infected[split_index:]

model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

mae = np.mean(np.abs(test_predictions - y_test))
print("Mean Absolute Error:", round(mae, 2))
