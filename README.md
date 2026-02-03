# MediNetNode

![Version](https://img.shields.io/badge/version-0.2-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-orange.svg)
![Security](https://img.shields.io/badge/security-61%25%20complete-green.svg)
>[!WARNING]
>This software is still in development (v0.2)

## Overview

MediNetNode is a software that complements MedinetHub by providing a way to a centers manage their datasets and people who can acces, allow users to train their own models using the datasets uploaded to the Node and to monitor the models during training without sharing data.

## Features Available

- ✅ Manage and upload datasets
- ✅ Create and delete permissions for users to access and train on the datasets
- ✅ Models will always train with Differential Privacy to ensure security and more anonymization
- ✅ Three levels of permisions:
  - Admin: Can do everything
  - Researcher: Can just access to the platform using API, can train models using the datasets uploaded to the Node if they have the necessary permissions
  - Auditor: Can access to the platform and audit all the logs

## Future Features

- 🔄 Project creation: Assign a group of datasets to a specific group of people for a given date.
- 🔄 SVM and Random Forest federated with Differential Privacy (In progress)
- 🔄 More parameters to configure in your models like learning rate scheduler and other callbacks
- 🔄 Inference from pre-trained models. The researcher can send data to run a model directly to the node without needing to train and download it using API.

## Security Status

**MediNetNode v0.2 is SECURE for development environments.**

- ✅ 11/23 vulnerabilities from security audit corrected (48%)
- ✅ 3 additional security features implemented
- ✅ 54/54 security tests passing (100%)
- ✅ **Real status: 14/23 (61%)**

Remaining vulnerabilities are production-specific (2FA, HTTPS, Email verification, etc.).

For full security details, see [SECURITY.md](MediNetNode-main/SECURITY.md)

## Links

- **For a Researchers**: If you are a researcher, please check [MediNet-Hub](https://github.com/isglobal-brge/MediNetHub)
- **Documentation**: [Medinet Documentation](https://isglobal-brge.github.io/MediNet/index.html)
- **Application Note Paper**: Coming soon

## Contact

- **Ramon Mateo Navarro**: ramon.mateo@citm.upc.edu
- **Juan R. González, PhD (Supervisor)**: juanr.gonzalez@isglobal.org

---

*Developed at ISGlobal*
