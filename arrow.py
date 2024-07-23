import matplotlib.pyplot as plt

# Membuat plot
fig, ax = plt.subplots()

# Menambahkan garis dengan panah
ax.annotate('', xy=(0.8, 0.2), xytext=(0.2, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

# Menambahkan garis dengan panah lainnya
ax.annotate('', xy=(0.2, 0.2), xytext=(0.8, 0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))

# Menambahkan titik untuk referensi
ax.plot([0.2, 0.8], [0.8, 0.2], 'ro')  # Titik (0.2, 0.8) dan (0.8, 0.2)
ax.plot([0.8, 0.2], [0.8, 0.2], 'ro')  # Titik (0.8, 0.8) dan (0.2, 0.2)


# Menampilkan plot
plt.title('Garis dengan Panah Menggunakan Matplotlib')
plt.savefig(f'cek_arraow.png')
plt.close()