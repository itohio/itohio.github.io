---
title: "DNS Fundamentals & Configuration"
date: 2024-12-12
draft: false
category: "net"
tags: ["dns", "networking", "gmail", "github-pages", "email"]
---

DNS fundamentals and practical configuration for common services like Gmail and GitHub Pages.

---

## DNS Basics

### DNS Record Types

| Record | Purpose | Example |
|--------|---------|---------|
| **A** | IPv4 address | `example.com → 192.0.2.1` |
| **AAAA** | IPv6 address | `example.com → 2001:db8::1` |
| **CNAME** | Alias to another domain | `www.example.com → example.com` |
| **MX** | Mail server | `example.com → mail.example.com` |
| **TXT** | Text records (SPF, DKIM, verification) | `"v=spf1 include:_spf.google.com ~all"` |
| **NS** | Name servers | `example.com → ns1.provider.com` |
| **SOA** | Start of authority | Zone metadata |
| **SRV** | Service locator | `_service._proto.name` |
| **CAA** | Certificate authority authorization | `0 issue "letsencrypt.org"` |
| **PTR** | Reverse DNS | `1.2.0.192.in-addr.arpa → example.com` |

---

## DNS Lookup Tools

### dig (Recommended)

```bash
# Basic lookup
dig example.com

# Specific record type
dig example.com A
dig example.com AAAA
dig example.com MX
dig example.com TXT
dig example.com NS

# Short answer only
dig example.com +short

# Query specific nameserver
dig @8.8.8.8 example.com

# Reverse DNS lookup
dig -x 192.0.2.1

# Trace DNS resolution path
dig example.com +trace

# Show all records
dig example.com ANY
```

### nslookup

```bash
# Basic lookup
nslookup example.com

# Specific record type
nslookup -type=A example.com
nslookup -type=MX example.com
nslookup -type=TXT example.com

# Query specific nameserver
nslookup example.com 8.8.8.8
```

### host

```bash
# Basic lookup
host example.com

# Specific record type
host -t A example.com
host -t MX example.com
host -t TXT example.com

# Verbose output
host -v example.com
```

---

## Gmail/Google Workspace DNS Configuration

### MX Records (Mail Routing)

Priority matters - lower number = higher priority.

```
Priority  Hostname
1         aspmx.l.google.com
5         alt1.aspmx.l.google.com
5         alt2.aspmx.l.google.com
10        alt3.aspmx.l.google.com
10        alt4.aspmx.l.google.com
```

**DNS Configuration**:
```
Type: MX
Name: @
Value: 1 aspmx.l.google.com.
TTL: 3600

Type: MX
Name: @
Value: 5 alt1.aspmx.l.google.com.
TTL: 3600

Type: MX
Name: @
Value: 5 alt2.aspmx.l.google.com.
TTL: 3600

Type: MX
Name: @
Value: 10 alt3.aspmx.l.google.com.
TTL: 3600

Type: MX
Name: @
Value: 10 alt4.aspmx.l.google.com.
TTL: 3600
```

### SPF Record (Sender Policy Framework)

Prevents email spoofing by specifying authorized mail servers.

```
Type: TXT
Name: @
Value: v=spf1 include:_spf.google.com ~all
TTL: 3600
```

**SPF Syntax**:
- `v=spf1`: SPF version 1
- `include:_spf.google.com`: Include Google's SPF records
- `~all`: Soft fail for others (mark as spam but accept)
- `-all`: Hard fail for others (reject)
- `+all`: Allow all (NOT recommended)

### DKIM Record (DomainKeys Identified Mail)

Cryptographic signature to verify email authenticity.

```
Type: TXT
Name: google._domainkey
Value: v=DKIM1; k=rsa; p=MIGfMA0GCSqGSIb3DQEBAQUAA4GN...
TTL: 3600
```

**Get your DKIM key**:
1. Go to Google Admin Console
2. Apps → Google Workspace → Gmail → Authenticate email
3. Generate new record
4. Copy the TXT record value

### DMARC Record (Domain-based Message Authentication)

Policy for handling failed SPF/DKIM checks.

```
Type: TXT
Name: _dmarc
Value: v=DMARC1; p=quarantine; rua=mailto:dmarc@example.com
TTL: 3600
```

**DMARC Policies**:
- `p=none`: Monitor only (no action)
- `p=quarantine`: Mark as spam
- `p=reject`: Reject email
- `rua=mailto:...`: Aggregate reports
- `ruf=mailto:...`: Forensic reports
- `pct=100`: Apply policy to 100% of emails

### Verification TXT Record

Google requires verification before using Gmail.

```
Type: TXT
Name: @
Value: google-site-verification=abc123xyz...
TTL: 3600
```

### Complete Gmail DNS Example

```dns
; MX Records
@    IN MX 1  aspmx.l.google.com.
@    IN MX 5  alt1.aspmx.l.google.com.
@    IN MX 5  alt2.aspmx.l.google.com.
@    IN MX 10 alt3.aspmx.l.google.com.
@    IN MX 10 alt4.aspmx.l.google.com.

; SPF Record
@    IN TXT "v=spf1 include:_spf.google.com ~all"

; DKIM Record
google._domainkey IN TXT "v=DKIM1; k=rsa; p=YOUR_PUBLIC_KEY"

; DMARC Record
_dmarc IN TXT "v=DMARC1; p=quarantine; rua=mailto:dmarc@example.com"

; Verification
@    IN TXT "google-site-verification=YOUR_VERIFICATION_CODE"
```

---

## GitHub Pages DNS Configuration

### Custom Domain (Apex Domain)

For `example.com`:

```
Type: A
Name: @
Value: 185.199.108.153
TTL: 3600

Type: A
Name: @
Value: 185.199.109.153
TTL: 3600

Type: A
Name: @
Value: 185.199.110.153
TTL: 3600

Type: A
Name: @
Value: 185.199.111.153
TTL: 3600
```

**All 4 A records are required** for redundancy and load balancing.

### Custom Subdomain (www)

For `www.example.com`:

```
Type: CNAME
Name: www
Value: yourusername.github.io.
TTL: 3600
```

**Note**: The trailing dot (`.`) is important!

### Both Apex and www

```dns
; Apex domain (example.com)
@    IN A 185.199.108.153
@    IN A 185.199.109.153
@    IN A 185.199.110.153
@    IN A 185.199.111.153

; www subdomain (www.example.com)
www  IN CNAME yourusername.github.io.
```

### Verification (Optional but Recommended)

```
Type: TXT
Name: _github-pages-challenge-yourusername
Value: verification-code-from-github
TTL: 3600
```

### Complete GitHub Pages Example

```dns
; GitHub Pages A records
@    IN A 185.199.108.153
@    IN A 185.199.109.153
@    IN A 185.199.110.153
@    IN A 185.199.111.153

; www subdomain
www  IN CNAME yourusername.github.io.

; Verification (if required)
_github-pages-challenge-yourusername IN TXT "verification-code"
```

### GitHub Pages Configuration

After DNS setup:

1. Go to repository Settings → Pages
2. Enter custom domain: `example.com` or `www.example.com`
3. Wait for DNS check (can take 24-48 hours)
4. Enable "Enforce HTTPS" (after DNS propagates)

---

## Combined Example: Gmail + GitHub Pages

```dns
; GitHub Pages
@    IN A     185.199.108.153
@    IN A     185.199.109.153
@    IN A     185.199.110.153
@    IN A     185.199.111.153
www  IN CNAME yourusername.github.io.

; Gmail MX Records
@    IN MX 1  aspmx.l.google.com.
@    IN MX 5  alt1.aspmx.l.google.com.
@    IN MX 5  alt2.aspmx.l.google.com.
@    IN MX 10 alt3.aspmx.l.google.com.
@    IN MX 10 alt4.aspmx.l.google.com.

; Email Authentication
@              IN TXT "v=spf1 include:_spf.google.com ~all"
google._domainkey IN TXT "v=DKIM1; k=rsa; p=YOUR_DKIM_KEY"
_dmarc         IN TXT "v=DMARC1; p=quarantine; rua=mailto:dmarc@example.com"

; Verification
@    IN TXT "google-site-verification=YOUR_GOOGLE_CODE"
_github-pages-challenge-yourusername IN TXT "YOUR_GITHUB_CODE"
```

---

## DNS Propagation & Testing

### Check DNS Propagation

```bash
# Check from multiple locations
# Use online tools:
# - https://dnschecker.org
# - https://www.whatsmydns.net

# Check locally
dig example.com @8.8.8.8
dig example.com @1.1.1.1
dig example.com @your-isp-dns
```

### Test Email Configuration

```bash
# Check MX records
dig example.com MX +short

# Check SPF
dig example.com TXT +short | grep spf

# Check DKIM
dig google._domainkey.example.com TXT +short

# Check DMARC
dig _dmarc.example.com TXT +short
```

### Test Email Deliverability

Online tools:
- **MXToolbox**: https://mxtoolbox.com
- **Google Admin Toolbox**: https://toolbox.googleapps.com/apps/checkmx/
- **Mail-tester**: https://www.mail-tester.com

### Flush DNS Cache

```bash
# Linux (systemd-resolved)
sudo systemd-resolve --flush-caches

# macOS
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder

# Windows
ipconfig /flushdns
```

---

## Common DNS Providers

### Cloudflare

```bash
# API example (set A record)
curl -X POST "https://api.cloudflare.com/client/v4/zones/ZONE_ID/dns_records" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"type":"A","name":"example.com","content":"192.0.2.1","ttl":3600}'
```

### AWS Route 53

```bash
# CLI example
aws route53 change-resource-record-sets --hosted-zone-id ZONE_ID --change-batch '{
  "Changes": [{
    "Action": "CREATE",
    "ResourceRecordSet": {
      "Name": "example.com",
      "Type": "A",
      "TTL": 300,
      "ResourceRecords": [{"Value": "192.0.2.1"}]
    }
  }]
}'
```

### Google Cloud DNS

```bash
# gcloud example
gcloud dns record-sets create example.com. \
  --zone=my-zone \
  --type=A \
  --ttl=300 \
  --rrdatas=192.0.2.1
```

---

## DNS Security

### DNSSEC (DNS Security Extensions)

```bash
# Check DNSSEC validation
dig example.com +dnssec

# Check DS records
dig example.com DS +short
```

### CAA Records (Certificate Authority Authorization)

```
Type: CAA
Name: @
Value: 0 issue "letsencrypt.org"
TTL: 3600

Type: CAA
Name: @
Value: 0 issuewild "letsencrypt.org"
TTL: 3600

Type: CAA
Name: @
Value: 0 iodef "mailto:security@example.com"
TTL: 3600
```

---

## Troubleshooting

### Email Not Working

```bash
# 1. Check MX records
dig example.com MX +short

# 2. Check SPF
dig example.com TXT +short | grep spf

# 3. Test with mail-tester.com
# Send email to the provided address

# 4. Check Google Admin Console
# Apps → Google Workspace → Gmail → Authenticate email
```

### GitHub Pages Not Loading

```bash
# 1. Check A records
dig example.com +short

# Should return all 4 GitHub IPs:
# 185.199.108.153
# 185.199.109.153
# 185.199.110.153
# 185.199.111.153

# 2. Check CNAME (if using www)
dig www.example.com +short

# Should return: yourusername.github.io

# 3. Wait for propagation (up to 48 hours)

# 4. Check GitHub Pages settings
# Repository → Settings → Pages
```

### DNS Not Propagating

```bash
# Check TTL (Time To Live)
dig example.com | grep "^example.com"

# Lower TTL before making changes
# Wait for old TTL to expire
# Make changes
# Increase TTL again
```

---

## Quick Reference

### Gmail DNS Records

```
MX:   1  aspmx.l.google.com.
MX:   5  alt1.aspmx.l.google.com.
MX:   5  alt2.aspmx.l.google.com.
MX:   10 alt3.aspmx.l.google.com.
MX:   10 alt4.aspmx.l.google.com.
TXT:  v=spf1 include:_spf.google.com ~all
TXT:  (DKIM at google._domainkey)
TXT:  (DMARC at _dmarc)
```

### GitHub Pages DNS Records

```
A:    185.199.108.153
A:    185.199.109.153
A:    185.199.110.153
A:    185.199.111.153
CNAME: www → yourusername.github.io.
```

---

## Tips

- **Always use trailing dots** in DNS records (e.g., `example.com.`)
- **Lower TTL before changes** to speed up propagation
- **Test with multiple DNS servers** (8.8.8.8, 1.1.1.1, etc.)
- **Wait 24-48 hours** for full DNS propagation
- **Use `dig +short`** for quick checks
- **Enable DNSSEC** for security (if provider supports it)
- **Set up DMARC** to monitor email authentication
- **Use CAA records** to restrict certificate issuance
- **Test email deliverability** with mail-tester.com
- **Keep verification TXT records** even after verification

