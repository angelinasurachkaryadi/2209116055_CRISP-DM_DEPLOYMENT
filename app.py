import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Supermarket Sales Analysis and Prediction")

st.write('memberikan gambaran komprehensif tentang penjualan supermarket menawarkan wawasan tentang perilaku pembelian konsumen,tren produk,dan dinamika ritel.Menganalisis data ini memungkinkan pengecer,pemasar,dan analisis untuk mengoptimalkan strategi,meningkatkan manajemen inventaris,dan meningkatkan pengalaman pelanggan..')

df = pd.read_csv('supermarket_sales.csv')

st.write(df)

st.subheader('Survival Rate')
survival_counts = df['Customer type'].value_counts()
st.text(f'Survival Rate {survival_counts.values[1] / sum(survival_counts):.2%}')

fig1, ax1 = plt.subplots()
survival_counts.plot.bar(ax=ax1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
df['Quantity'].plot.hist(ax=ax2)
st.pyplot(fig2)

# Visualisasi Histogram Unit Price
st.subheader('Histogram Unit Price')
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.hist(df['Unit price'], bins=20, color='skyblue', edgecolor='black')
ax3.set_xlabel('Unit Price')
ax3.set_ylabel('Frequency')
ax3.set_title('Histogram of Unit Price')
st.pyplot(fig3)

# Plot pie chart of customer type distribution
st.subheader('Customer Type Distribution')
fig4, ax4 = plt.subplots()
df['Customer type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax4)
ax4.axis('equal')
st.pyplot(fig4)

# Tampilkan Histogram Rating
st.subheader("Histogram Rating")
fig, ax = plt.subplots()
sns.histplot(df['Rating'].dropna(), bins=10, kde=True, ax=ax)
st.pyplot(fig)

# Split data into features and target
X = df[['Customer type', 'Unit price', 'Quantity', 'Rating']]
y = df['Customer type']

# Encode categorical feature 'Customer type'
le = LabelEncoder()
X['Customer type'] = le.fit_transform(X['Customer type'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, 'gnb.pkl')

def predict(customer_type, unit_price, quantity, rating):
    input_data = pd.DataFrame([[customer_type, unit_price, quantity, rating]], columns=['Customer type', 'Unit price', 'Quantity', 'Rating'])
    prediction = clf.predict(input_data)
    return prediction[0]


st.subheader("Making Prediction")

customer_type_option = [1, 0]  # 1 for 'Member', 0 for 'Normal'
customer_type = st.selectbox('Customer type', customer_type_option)

unit_price = st.number_input('Unit price', min_value=0.0)
quantity = st.number_input('Quantity', min_value=1)
rating = st.number_input('Rating', min_value=0.0, max_value=10.0, step=0.1)

if st.button('Predict'):
    prediction = predict(customer_type, unit_price, quantity, rating)
    if prediction == 1:
        predicted_type = 'Member'
    elif prediction == 0:
        predicted_type = 'Normal'
    else:
        predicted_type = 'Unknown'
    st.write(f'Predicted Customer Type: {predicted_type}')

# Interpretation and actionable insight
st.subheader("Interpretation and Actionable Insight")

interpretation = "Dari hasil prediksi, terlihat bahwa sebagian besar prediksi jenis pelanggan supermarket adalah 'Normal' daripada 'Member', dengan nilai lebih dari 50% untuk 'Normal'."
insight = "Ini menunjukkan bahwa pola pembelian pelanggan cenderung lebih mirip dengan pelanggan biasa daripada pelanggan yang memiliki keanggotaan khusus."
actionable_insight = "Memperhatikan pola pembelian ini, strategi promosi dapat difokuskan pada menarik lebih banyak pelanggan biasa. Misalnya, menawarkan diskon khusus untuk pembelian dalam jumlah tertentu atau membuat program loyalitas untuk meningkatkan jumlah pembelian pelanggan biasa. Selain itu, penawaran spesial pada produk tertentu yang sering dibeli oleh pelanggan biasa dapat meningkatkan minat pembelian dan meningkatkan pendapatan toko."

st.write(interpretation)
st.write(insight)
st.write(actionable_insight)








