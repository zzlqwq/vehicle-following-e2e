# Vehicle Following E2E

This repository provides the code and datasets corresponding to the paper:

**All You Need Is One Camera: An End-to-End Autonomous Driving Framework for Vehicle Following**

## Abstract
The rapid increase in vehicle ownership has exacerbated traffic congestion, accidents, and carbon emissions. Vehicle platooning offers a promising solution but often relies on structured environments and costly sensors. To address these limitations, we propose a cost-effective, generalizable end-to-end vehicle-following framework using only a monocular fisheye camera. This method tackles key challenges, including:

- Mitigating causal confusion through semantic masking in BEV networks.
- Enhancing trajectory inference with a dynamic sampling mechanism for spatio-temporal data.
- Conducting real-world validations, demonstrating improved performance over traditional systems in diverse conditions.

Our approach unifies perception and control within a single framework, achieving robust vehicle-following performance in complex and unstructured scenarios.

## Repository Overview

### Folder Structure
- **`model/`**: Implementation of the end-to-end vehicle-following framework.
- **`docs/`**: Documentation and tutorials.

### Features
- Code for semantic mask and dynamic sampling.
- Preprocessing and visualization tools for fisheye camera inputs.
- End-to-end pipeline from semantic mask to trajectory planning.

## Getting Started

### Installation
Clone the repository and navigate to the folder:

```bash
git clone https://github.com/yourusername/vehicle-following-e2e.git
cd vehicle-following-e2e
```

### Dataset
The dataset used in this project includes various scenarios. Only a partial dataset has been uploaded and is available on Google Drive. You can access it [here](https://drive.google.com/drive/folders/1_GenfbRosUPFyHhQkU8nwK0usfB_Rvzf?usp=drive_link).

## Demo Video
A demonstration of the proposed framework in action is available on [YouTube](https://youtu.be/zL1bcVb9kqQ).

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
We would like to thank the contributors and community members who supported the development of this project.

