# Azure Machine Learning

## Ensure Permissions

1. Use Azure PIM to Activate Owner role for Deep Learning Group subscription for your SC-Alt account.
This should implicitly give you access to Reuben's workspace in the subscription

2. On `yuwanechelon` storage account, Assign "Storage Blob Data Contributor" and "Storage Table Data Contributor" roles to your SC-Alt account

### Set Env Vars

```bash
# Project
export ALIAS=mattm
export PROJECT_NAME="echelon-$ALIAS"

# Weights and Biases
export WANDB_BASE_URL=https://microsoft-research.wandb.io
export WANDB_PROJECT=project-echelon

# Subscriptions
export PROJECT_ECHELON_SUBSCRIPTION_ID=$(az account list --query "[?name=='Project Echelon'].{id:id}[0]" -o tsv)
export DEEP_LEARNING_GROUP_SUBSCRIPTION_ID=$(az account list --query "[?name=='Deep Learning Group'].{id:id}[0]" -o tsv)

# AML
export AML_SUBSCRIPTION_ID=$DEEP_LEARNING_GROUP_SUBSCRIPTION_ID
export AML_RESOURCE_GROUP_NAME="Reuben-Tan"
export AML_WORKSPACE_NAME="reubenws"
export AML_MANAGED_IDENTITY="reubenuai"
export AML_MANAGED_IDENTITY_RESOURCE_URI="/subscriptions/2cd190bb-b42a-477c-b1bb-2f20932d8dc5/resourcegroups/Reuben-Tan/providers/Microsoft.ManagedIdentity/userAssignedIdentities/reubenuai"

# ACR
export ACR_NAME="reubencr"
export ACR_RESOURCE_GROUP_NAME="Reuben-Tan"

# Storage
export STORAGE_RESOURCE_GROUP_NAME="yuwan-echelon"
export STORAGE_ACCOUNT_NAME="yuwanechelon"
export STORAGE_CONTAINER_NAME="data"
```

### Submit Job

```bash
az ml job create -f aml/finetune-openpi-echelon-aml.yaml \
--resource-group $AML_RESOURCE_GROUP_NAME \
--workspace-name $AML_WORKSPACE_NAME \
--subscription $AML_SUBSCRIPTION_ID \
--query "services.Studio.endpoint" \
-o tsv
```

## Other (Amulet)

### Add storage credential

```bash
amlt cred storage set $STORAGE_ACCOUNT_NAME \
--subscription $DEEP_LEARNING_GROUP_SUBSCRIPTION_ID \
--resource-group $STORAGE_RESOURCE_GROUP_NAME
```

### Create Project

```bash
amlt project create $PROJECT_NAME $STORAGE_ACCOUNT_NAME
```

## If project is already created, checkout the existing project

```bash
amlt project checkout $PROJECT_NAME $STORAGE_ACCOUNT_NAME
```

### Add GCR Default Targets

```bash
amlt target add-defaults gcr
```

### Add target

```bash
amlt target add \
--subscription $AML_SUBSCRIPTION_ID \
--resource-group $AML_RESOURCE_GROUP_NAME \
--workspace-name $AML_WORKSPACE_NAME
```

### Add Workspace

```bash
amlt workspace add \
--subscription $AML_SUBSCRIPTION_ID \
--resource-group $AML_RESOURCE_GROUP_NAME \
$AML_WORKSPACE_NAME
```

### Submit Job

```bash
amlt run aml/finetune-openpi-echelon-amulet.yaml echelon-openpi-finetune --yes -d ""
```

## Other Notes

### Resources

https://amulet-docs.azurewebsites.net/
https://dev.azure.com/dlmm/_git/ProjectWillow?path=/amlt/README.md

### Useful Commands

#### Get target SKUs by type

```bash
amlt cache instance-types -I A100
amlt cache instance-types -I MI300
```
