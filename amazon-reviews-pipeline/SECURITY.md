# Security Best Practices

## API Key Management

This project requires an OpenAI API key to function. Follow these security best practices:

### ✅ DO:

1. **Use Environment Variables**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. **Use .env Files (for local development only)**
   - Copy `.env.example` to `.env`
   - Add your API key to `.env`
   - Never commit `.env` to version control

3. **Use Secrets Management in Production**
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
   - Kubernetes Secrets

4. **Rotate Keys Regularly**
   - Set up key rotation schedules
   - Monitor key usage

5. **Set Usage Limits**
   - Configure spending limits in OpenAI dashboard
   - Monitor API usage regularly

### ❌ DON'T:

1. **Never hardcode API keys in source code**
   ```python
   # WRONG - Never do this!
   api_key = "sk-proj-xxxxx"
   ```

2. **Never commit API keys to version control**
   - Check files before committing
   - Use git hooks to prevent accidental commits

3. **Never share API keys**
   - Each developer should use their own key
   - Use separate keys for different environments

4. **Never expose keys in logs or error messages**
   - Sanitize logs before sharing
   - Use secure logging practices

## If a Key is Exposed

1. **Immediately revoke the key** in the OpenAI dashboard
2. **Generate a new key**
3. **Update all applications** using the old key
4. **Review logs** for unauthorized usage
5. **Notify your team** about the incident

## Additional Security Measures

1. **Use .gitignore**
   - Ensure sensitive files are excluded
   - Review `.gitignore` regularly

2. **Code Reviews**
   - Always review for hardcoded secrets
   - Use automated scanning tools

3. **Access Control**
   - Limit who has access to production keys
   - Use principle of least privilege

4. **Monitoring**
   - Set up alerts for unusual API usage
   - Regular security audits

## Resources

- [OpenAI API Key Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [OWASP Secrets Management](https://owasp.org/www-project-secrets-management/)
- [12 Factor App - Config](https://12factor.net/config)