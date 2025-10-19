from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


tabular_data = pd.read_csv('student_exam_scores.csv')
tabular_data = tabular_data.drop('student_id', axis=1)
# tabular_data.info()

column_names = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']

X = tabular_data[column_names]
y = tabular_data['exam_score']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)

tabular_data['predicted_exam_score'] = model.predict(X)

# tabular_data.to_csv("table_with_predicted_exam_score.csv", index=False)


mse = mean_squared_error(y, tabular_data['predicted_exam_score'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, tabular_data['predicted_exam_score'])
r2 = r2_score(y, tabular_data['predicted_exam_score'])

print(f"Метрики качества модели:")
print(f"R² (коэффициент детерминации): {r2:.4f}")
print(f"MSE (средняя квадратичная ошибка): {mse:.4f}")
print(f"RMSE (среднеквадратичная ошибка): {rmse:.4f}")
print(f"MAE (средняя абсолютная ошибка): {mae:.4f}")
print(f"\nКоэффициенты модели:")


model_coeffs = []
for feature, coef in zip(column_names, model.coef_):
    model_coeffs.append(coef)

def normalize_coeffs(your_list):
    norm_c = []
    for x in your_list:
        n = (x / sum(your_list)) * 100
        norm_c.append(float(n))
    return norm_c

norm_coeffs = normalize_coeffs(model_coeffs)
for feature, coef, per in zip(column_names, model.coef_, norm_coeffs):
    print(f"{feature}: {coef:.4f} ({round(per, 2)}%)")



plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.figure(figsize=(12, 7))
plt.plot(y.values, label='Фактические', alpha=0.7)
plt.plot(tabular_data['predicted_exam_score'].values, label='Предсказанные', alpha=0.7)
plt.xlabel('Наблюдение')
plt.ylabel('Exam Score')
plt.title('Фактические и предсказанные значения')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Влияние каждого признака на предсказания
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# Влияние часов учебы на предсказания
axes[0, 0].scatter(tabular_data['hours_studied'], y, alpha=0.5, label='Фактические')
axes[0, 0].scatter(tabular_data['hours_studied'], tabular_data['predicted_exam_score'], alpha=0.5, label='Предсказанные')
axes[0, 0].set_xlabel('Hours Studied')
axes[0, 0].set_ylabel('Exam Score')
axes[0, 0].set_title('Влияние часов учебы')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Влияние часов сна на предсказания
axes[0, 1].scatter(tabular_data['sleep_hours'], y, alpha=0.5, label='Фактические')
axes[0, 1].scatter(tabular_data['sleep_hours'], tabular_data['predicted_exam_score'], alpha=0.5, label='Предсказанные')
axes[0, 1].set_xlabel('Sleep Hours')
axes[0, 1].set_ylabel('Exam Score')
axes[0, 1].set_title('Влияние часов сна')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Влияние посещаемости на предсказания
axes[1, 0].scatter(tabular_data['attendance_percent'], y, alpha=0.5, label='Фактические')
axes[1, 0].scatter(tabular_data['attendance_percent'], tabular_data['predicted_exam_score'], alpha=0.5, label='Предсказанные')
axes[1, 0].set_xlabel('Attendance Percent')
axes[1, 0].set_ylabel('Exam Score')
axes[1, 0].set_title('Влияние посещаемости')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Влияние предыдущих оценок на предсказания
axes[1, 1].scatter(tabular_data['previous_scores'], y, alpha=0.5, label='Фактические')
axes[1, 1].scatter(tabular_data['previous_scores'], tabular_data['predicted_exam_score'], alpha=0.5, label='Предсказанные')
axes[1, 1].set_xlabel('Previous Scores')
axes[1, 1].set_ylabel('Exam Score')
axes[1, 1].set_title('Влияние предыдущих оценок')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()