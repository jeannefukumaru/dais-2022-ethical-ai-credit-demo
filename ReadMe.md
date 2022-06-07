## Connecting Databricks Model Registry Webhooks to GitHub Actions

The guide shows how you can use Azure Function as an intermediate service to enable Databricks Model Registry to trigger a GitHub Actions workflow

1. Clone this repo and deploy to Azure Function. You can Read [this tutorial](https://docs.microsoft.com/en-us/azure/azure-functions/create-first-function-vs-code-csharp?tabs=in-process) to get started with Azure Function.
2. Modify the code in `HttpExample/__init__py` to specify the trigger API call for GitHub Actions.
3. Configure a webhook in databricks to trigger the HTTP URL of your function app. Please make sure that you append your function name at the end of the URL (`HttpExample` for this repo). The complete URL will look something like this: `https://{FUNCTION_APP_NAME}.net/api/HttpExample`
4. [optional] Add authentication to your Azure Function. You can follow [this guide](https://docs.microsoft.com/en-us/azure/app-service/configure-authentication-provider-aad#-configure-with-express-settings) to configure authentication for Azure Function. 
5. Create a GitHub actions workflow that is triggerable by API. You can read more on it [here](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#repository_dispatch). This repo contains a sample workflow in `.github` folder that posts a comment to model registrY on recieving the webhook.
