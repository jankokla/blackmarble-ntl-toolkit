---
icon: lucide/table
---

# Required Products & Bands

Here is a summary of all available preprocessing methods and the satellite products and bands they require.

| Method | Required Products & Bands |
| ------ | ------------------------- |
| `BlackMarbleHighQualityFilter` | **VNP46A2:** `DNB_BRDF_Corrected_NTL`, `Mandatory_Quality_Flag` |
| `FilterLowNTL` | **VNP46A2:** `DNB_BRDF_Corrected_NTL` |
| `Jia2023HighQualityFilter` | **VNP46A1:** `Solar_Zenith`, `Moon_Illumination_Fraction`<br>**VNP46A2:** `DNB_BRDF_Corrected_NTL`, `Mandatory_Quality_Flag`, `QF_Cloud_Mask` |
| `CloudSnowFilter` | **VNP46A2:** `DNB_BRDF_Corrected_NTL`, `QF_Cloud_Mask`, `Snow_Flag` |
| `ModifiedZScoreOutlierRemoval` | **VNP46A2:** `DNB_BRDF_Corrected_NTL` |
| `QuadraticVZACorrection` | **VNP46A1:** `Sensor_Zenith`<br>**VNP46A2:** `DNB_BRDF_Corrected_NTL` |
| `Hu2024AngularCorrection` | **VNP46A2:** `DNB_BRDF_Corrected_NTL`<br>**NOAA/VIIRS/DNB/ANNUAL_V22:** `average` |
| `AveragePooling2D` | **VNP46A2:** `DNB_BRDF_Corrected_NTL` |
| `Hu2024AAveraging` | **VNP46A2:** `DNB_BRDF_Corrected_NTL` |
| `LinearInterpolationGapFilling` | **VNP46A2:** `DNB_BRDF_Corrected_NTL` |
| `Yue2026DisturbanceFactorCorrection` | **VNP46A1:** `Sensor_Zenith`, `Sensor_Azimuth`<br>**VNP46A2:** `DNB_BRDF_Corrected_NTL`, `DNB_Lunar_Irradiance`, `QF_Cloud_Mask` |
