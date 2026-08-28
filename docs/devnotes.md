# Developer Notes

## Design Decisions  
Document important decisions about your project's architecture, algorithms, or methodologies:

### Choice of RECIST longest diameter calculation algorithm
[2026-08-10] Katy S. compared the different algorithms developed by MedSAM2 and BHKLab teams to calculate the longest axial diameter of a 3D segmentation. MedSAM2's method in [`compute_recist_line`](../src/ab_testing/utils/conver_nifti_to_recist_npz.py) under function  utilizes OpenCV contours. BHKLab's method (by Kaitlyn K.) in [`get_recist_pts`](../src/ab_testing/utils/annotations.py) measures from the centroid of the slice based on a `transformation of the origin * half of the major axis length`. Based on some testing with the NSCLC-Radiomics dataset, the MedSAM2 method was visually assessed to be more accurate. [Exploration script](../sandbox/recist_comparison.py)

### Multiple lesion handling  
[2026-08-10] The nifti to recist npz setup right now expects an nnUnet directory structure for images where the label files can contain multiple lesions with each lesion having a different label number. I (Katy) typically run med-imagetools with the SEPARATE ROI strategy, but could use MERGE so that the data is set up this way. 


## Technical Challenges  
Record significant problems you encountered and how you solved them


## Dependencies and Environment

Document specific version requirements or compatibility issues:

``` markdown
### Critical Version Dependencies  
[2025-04-25] SimpleITK 2.4.1 introduced a bug that flips images, so we froze version 2.4.0
```

## Best Practices

- Date your entries when appropriate
- Link to relevant code files or external resources
- Include small code snippets when helpful
- Note alternatives you considered and why they were rejected
- Document failed approaches to prevent others from repeating mistakes
- Update notes when major changes are made to the approach
