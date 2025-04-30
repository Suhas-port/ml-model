import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Build MesoNet model
def build_mesonet():
    input_layer = Input(shape=(256, 256, 3))

    x = Conv2D(8, (3, 3), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

# Build and Save Model
model = build_mesonet()
model.save('models/mesonet_model.h5')

print("✅ Model saved successfully at 'models/mesonet_model.h5'")
