{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Spoiler-xl/ransomware-detector/blob/main/uisteamnew.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 384
        },
        "id": "GfQ2JbAo_lqo",
        "outputId": "bb33d41e-eaf0-476a-97d3-202a175f46e6"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'streamlit'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-53f4138950fc>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mstreamlit\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mjoblib\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpefile\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtempfile\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import numpy as np\n",
        "import joblib\n",
        "import pefile\n",
        "import tempfile\n",
        "\n",
        "# Load model\n",
        "try:\n",
        "    with open(\"rf_model.pkl\", \"rb\") as file:\n",
        "        model = joblib.load(file)\n",
        "except FileNotFoundError:\n",
        "    st.error(\"Model file not found. Upload rf_model.pkl.\")\n",
        "    st.stop()\n",
        "\n",
        "st.title(\"🛡️ Ransomware Detection System\")\n",
        "st.markdown(\"Upload a PE file (.exe or .dll) to detect if it's ransomware or benign.\")\n",
        "\n",
        "uploaded_file = st.file_uploader(\"Upload PE File\", type=[\"exe\", \"dll\"])\n",
        "\n",
        "def extract_features(pe):\n",
        "    return [\n",
        "        pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress,  # DebugRVA\n",
        "        pe.FILE_HEADER.Machine,\n",
        "        pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,\n",
        "        pe.OPTIONAL_HEADER.MajorLinkerVersion,\n",
        "        pe.OPTIONAL_HEADER.DllCharacteristics,\n",
        "        pe.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress,  # IatVRA\n",
        "        pe.OPTIONAL_HEADER.MajorImageVersion\n",
        "    ]\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    with tempfile.NamedTemporaryFile(delete=False) as temp_file:\n",
        "        temp_file.write(uploaded_file.read())\n",
        "        temp_file_path = temp_file.name\n",
        "\n",
        "    try:\n",
        "        pe = pefile.PE(temp_file_path)\n",
        "        features = np.array([extract_features(pe)])\n",
        "        prediction = model.predict(features)[0]\n",
        "        proba = model.predict_proba(features)[0]\n",
        "\n",
        "        st.subheader(\"🔍 File Analysis Result\")\n",
        "\n",
        "        if prediction == 0:\n",
        "            st.error(\"🚨 Detected: RANSOMWARE\")\n",
        "            st.markdown(f\"**Confidence:** {proba[0]*100:.2f}% ransomware\")\n",
        "        else:\n",
        "            st.success(\"✅ Detected: BENIGN\")\n",
        "            st.markdown(f\"**Confidence:** {proba[1]*100:.2f}% benign\")\n",
        "\n",
        "        with st.expander(\"🔧 Extracted Features\"):\n",
        "            feature_names = [\n",
        "                \"DebugRVA\", \"Machine\", \"MajorOSVersion\",\n",
        "                \"MajorLinkerVersion\", \"DllCharacteristics\",\n",
        "                \"IatVRA\", \"MajorImageVersion\"\n",
        "            ]\n",
        "            for name, value in zip(feature_names, features[0]):\n",
        "                st.write(f\"**{name}**: {value}\")\n",
        "\n",
        "    except Exception as e:\n",
        "        st.error(f\"Error parsing PE file: {e}\")"
      ]
    }
  ]
}
