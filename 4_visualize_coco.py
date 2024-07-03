import numpy as np, os, random, json, matplotlib.pyplot as plt, matplotlib.patches as patches, cv2


def display_images_with_coco_ann(img_paths, annotations, display_type='both'):
    fig, axs= plt.subplots(2, 2, figsize=(10, 10))
    for ax, img_path in zip(axs.ravel(), img_paths):
        img= cv2.imread(img_path)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        ax.axis('off')
        
        img_filename= os.path.basename(img_path)
        
        #generator thats why we use next to get only one file it execution stop whenever find only one. 
        #we can use list but list is time consuming.
        img_id= next(item for item in annotations['images'] if item['file_name']==img_filename)['id']
        
        img_annotations= [ann for ann in annotations['annotations'] if ann['image_id']==img_id]
        
        colors=  [tuple(np.random.rand(3)) for _ in img_annotations]
        
        for ann, color in zip(img_annotations, colors):
            
            if display_type in ['bbox', 'both']:
                bbox= ann['bbox']
                rect= patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
                
                ax.add_patch(rect)
            
            if display_type in ['seg', 'both']:
                
                for seg in ann['segmentation']:
                    poly= [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    print()
                    print('---------color--',color)
                    print()
                    polygon= patches.Polygon(poly, closed=True, edgecolor=color, fill=False)
                    
                    ax.add_patch(polygon)
                
        
    plt.tight_layout()
    plt.show()

with open('COCO_output/train/coco_annotations.json', 'r') as f:
    annotations= json.load(f)
    
image_dir= 'COCO_output/train/'
all_image_files=[os.path.join(image_dir, img['file_name']) for img in annotations['images']]

random_image_files= random.sample(all_image_files, 4)
print(random_image_files,'---------------')
display_type='seg'

display_images_with_coco_ann(random_image_files, annotations, display_type)
