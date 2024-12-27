model=se3diff

#for dir in data openfold analysis model experiments
#  do
#    sed -i -e "s/import ${dir}/import ${model}_${dir}/g" */*.py
#    sed -i -e "s/import ${dir}/import ${model}_${dir}/g" */*/*.py
#    
#    sed -i -e "s/from ${dir}/from ${model}_${dir}/g" */*.py
#    sed -i -e "s/from ${dir}/from ${model}_${dir}/g" */*/*.py
#
#
#    sed -i -e "s/${dir}/${model}_${dir}/g" setup.py
#    sed -i -e "s/${dir}/${model}_${dir}/g" se3_diffusion.egg-info/*.txt
#
#    mv $dir "${model}_${dir}"
#  done
for dir in openfold
  do
    sed -i -e "s/import ${model}_${dir}/import ${dir}/g" */*.py
    sed -i -e "s/import ${model}_${dir}/import ${dir}/g" */*/*.py
    sed -i -e "s/import ${model}_${dir}/import ${dir}/g" */*/*/*.py
    
    sed -i -e "s/from ${model}_${dir}/from ${dir}/g" */*.py
    sed -i -e "s/from ${model}_${dir}/from ${dir}/g" */*/*/*.py


    sed -i -e "s/${model}_${dir}/${dir}/g" setup.py
    sed -i -e "s/${model}_${dir}/${dir}/g" se3_diffusion.egg-info/*.txt

    mv "${model}_${dir}" $dir
  done






